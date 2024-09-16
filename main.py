import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def prepare_training_data(df, output_file='training_data.txt'):
    """
    Подготавливает тренировочные данные в формате инструкции и ответа.

    Args:
        df (pd.DataFrame): Датафрейм с данными отзывов.
        output_file (str): Имя файла для сохранения тренировочных данных.
    """
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
    """
    Дообучает модель на предоставленных данных с использованием PEFT и LoRA.

    Args:
        model_name_or_path (str): Имя или путь к исходной модели.
        train_file (str): Путь к файлу с тренировочными данными.
        output_dir (str): Директория для сохранения дообученной модели.
        num_train_epochs (int): Количество эпох обучения.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = 0  # Устанавливаем pad_token_id

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=torch.float16,
        load_in_8bit=True  # Загружаем модель в 8-битном формате
    )

    # Подготовка модели для 8-битного обучения
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

    # Токенизация датасета
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,
            padding='max_length'
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Разделение на тренировочный и валидационный наборы
    tokenized_datasets = tokenized_datasets['train'].train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator
    )

    # Обучение модели
    trainer.train()

    # Сохранение дообученной модели
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def load_model(model_path):
    """
    Загружает дообученную модель и токенизатор.

    Args:
        model_path (str): Путь к дообученной модели.

    Returns:
        model: Загруженная модель.
        tokenizer: Загруженный токенизатор.
    """
    print(f"Loading model and tokenizer from {model_path}...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0  # Устанавливаем pad_token_id

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16,
        load_in_8bit=True
    )

    # Загружаем конфигурацию PEFT
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    print("Model loaded.")
    return model, tokenizer

def generate_answer(prompt, model, tokenizer, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """
    Генерирует ответ на заданный вопрос с использованием модели.

    Args:
        prompt (str): Вопрос или инструкция для модели.
        model: Модель для генерации ответа.
        tokenizer: Токенизатор для преобразования текста.
        max_new_tokens (int): Максимальное количество новых токенов.
        temperature (float): Параметр температуры для управления креативностью.
        top_p (float): Параметр nucleus sampling.

    Returns:
        str: Сгенерированный ответ.
    """
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=1024
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split('### Response:')[-1].strip()
    return answer

def main():
    # Загрузка данных
    df = pd.read_csv('processed_big.csv')  # Замените на имя вашего файла с данными
    print(f"Loaded {len(df)} rows of data.")

    # Подготовка тренировочных данных
    prepare_training_data(df, 'training_data.txt')

    # Дообучение модели
    fine_tune_model('llama-7b-hf', 'training_data.txt', 'fine-tuned-model', num_train_epochs=3)

    # Загрузка дообученной модели
    model, tokenizer = load_model('fine-tuned-model')

    # Вопросы
    questions = [
        "List the top 15 most frequently mentioned problems.",
        "List the top 15 most frequently mentioned bugs.",
        "List the top 15 features that users like the most."
    ]

    for question in questions:
        prompt = question
        answer = generate_answer(prompt, model, tokenizer)
        print(f"\nQ: {question}\nA: {answer}")

if __name__ == "__main__":
    main()
