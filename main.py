import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import pandas as pd
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def prepare_training_data(df, output_file='training_data.txt'):
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            positive_feedback = row['positive_feedback'] if pd.notna(row['positive_feedback']) else ''
            negative_feedback = row['negative_feedback'] if pd.notna(row['negative_feedback']) else ''

            if positive_feedback:
                instruction = "Extract the features that users like the most."
                output_text = positive_feedback.strip()
                f.write(f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}\n\n")

            if negative_feedback:
                instruction = "Extract the problems or bugs that users mention."
                output_text = negative_feedback.strip()
                f.write(f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}\n\n")
    print(f"Training data saved to {output_file}")


def fine_tune_model(model_name_or_path, train_file, output_dir, num_train_epochs=3):
    if os.path.exists(output_dir):
        print(f"Model already exists at {output_dir}. Skipping training.")
        return

    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',  # Распределение устройства автоматически
        torch_dtype=torch.float16,
        load_in_8bit=True
    )

    model = prepare_model_for_kbit_training(model)

    # Настройка параметров LoRA
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Применяем PEFT к модели
    model = get_peft_model(model, config)
    print("PEFT model prepared for training.")

    # Загрузка датасета
    dataset = load_dataset('text', data_files={'train': train_file})
    print("Training dataset prepared for training.")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,
            padding='max_length'
        )

    print("Tokenizer function prepared for training.")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )
    print("Tokenizer dataset prepared for training.")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    print("DataCollator function prepared for training.")
    tokenized_datasets = tokenized_datasets['train'].train_test_split(test_size=0.1)
    print("tokenized_datasets prepared for training.")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=16,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=500,
        learning_rate=5e-5,
        report_to='none',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    print("Training args prepared for training.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator
    )
    print("start training")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def load_model(model_path):
    print(f"Loading model and tokenizer from {model_path}...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        load_in_8bit_fp32_cpu_offload=True  # Включение выгрузки слоев на CPU
    )

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16,
        quantization_config=quantization_config  # Использование новой конфигурации для 8-битных моделей
    )

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def generate_answer(prompt, model, tokenizer, max_new_tokens=150, temperature=0.5, top_p=0.9):
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512  # Уменьшение длины
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        model.to('cuda')

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,  # Ограничение длины ответа
                do_sample=True,
                temperature=temperature,  # Снижение температуры для более точных ответов
                top_p=top_p,  # Nucleus sampling
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split('### Response:')[-1].strip()
    return answer


def load_untrained_model(model_path):
    """
    Функция загружает необученную модель и токенизатор из указанной директории.
    """
    print(f"Loading untrained model and tokenizer from {model_path}...")

    # Загрузка токенизатора
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0  # Установим pad_token_id на 0

    # Загрузка необученной модели
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map='auto',  # Используем GPU, если доступен
        torch_dtype=torch.float16  # Устанавливаем смешанную точность
    )

    model.eval()  # Переводим модель в режим inference (оценки)
    print("Model loaded.")

    return model, tokenizer


def generate_answer_untrained(prompt, model, tokenizer, max_new_tokens=150, temperature=0.7, top_p=0.9):
    """
    Генерация ответа на основе необученной модели
    """
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512  # Установим максимальную длину
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        model.to('cuda')

    with torch.cuda.amp.autocast():  # Используем AMP для повышения производительности
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,  # Устанавливаем температуру для управления случайностью
                top_p=top_p,  # Nucleus sampling
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split('### Response:')[-1].strip()
    return answer

def main():
    df = pd.read_csv('processed_big.csv')
    print(f"Loaded {len(df)} rows of data.")

    # Подготовка данных для обучения
    #prepare_training_data(df, 'training_data.txt')

    # Обучение модели, если она еще не обучена
    #fine_tune_model('llama-7b-hf', 'training_data.txt', 'fine-tuned-model', num_train_epochs=3)

    # Загрузка уже обученной модели
    #model, tokenizer = load_model('fine-tuned-model')
    model, tokenizer =load_untrained_model('llama-7b-hf')

    questions = [
        "Extract the features that users like the most.",
        "Extract the problems or bugs that users mention."
    ]

    for question in questions:
        prompt = question
        #answer = generate_answer(prompt, model, tokenizer)
        answer = generate_answer_untrained(prompt, model, tokenizer)
        print(f"\nQ: {question}\nA: {answer}")


if __name__ == "__main__":
    main()
