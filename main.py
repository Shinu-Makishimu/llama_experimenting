import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


def load_model(model_path):
    """
    Загружает обученную модель и токенизатор.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map='auto',  # Автоматическое распределение на устройства
        torch_dtype=torch.float16,
        load_in_8bit=True  # Загружаем модель в 8-битном формате
    )

    model.eval()
    return model, tokenizer


def generate_answer(prompt, model, tokenizer, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """
    Генерирует ответ на заданный вопрос с использованием обученной модели.
    """
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        # Убираем model.to('cuda'), т.к. модель уже правильно распределена

    with torch.cuda.amp.autocast():
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


def analyze_feedback(df, model, tokenizer):
    """
    Анализирует отзывы пользователей и генерирует список фич и багов.
    """
    # Объединяем все отзывы в одну строку
    positive_feedback_text = " ".join(df['positive_feedback'].dropna().tolist())
    negative_feedback_text = " ".join(df['negative_feedback'].dropna().tolist())

    # Генерация списка фич, которые нравятся пользователям
    prompt_positive = "List the top features that users like the most based on the following feedback: " + positive_feedback_text
    top_features = generate_answer(prompt_positive, model, tokenizer)

    # Генерация списка багов, которые мешают пользователям
    prompt_negative = "List the top bugs or problems that users mention based on the following feedback: " + negative_feedback_text
    top_bugs = generate_answer(prompt_negative, model, tokenizer)

    print("Top features users like the most:")
    print(top_features)

    print("\nTop bugs users mentioned the most:")
    print(top_bugs)


# Загрузка данных
df = pd.read_csv('processed_big.csv')

# Загрузка модели
model_path = 'fine-tuned-model'  # Путь к директории с обученной моделью
model, tokenizer = load_model(model_path)

# Анализ отзывов
analyze_feedback(df, model, tokenizer)
