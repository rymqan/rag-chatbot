from transformers import AutoModelForCausalLM, AutoTokenizer

from config.config import GENERATOR_MODEL, SYSTEM_PROMPT, MAX_NEW_TOKENS, DEVICE

model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL)
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)

model.to(DEVICE)

def generate_answer(query, context):
    # Combine the 
    combined_text = "\n\n".join(
        f"\n{source['text']}" for source in context
    )
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {"role": "user", "content": f"""
Вот информация из внутренних источников:\n\n
{combined_text}\n\n
Вопрос клиента: {query}\n
Ответ:
"""},
    ]

    # Tokenize and generate text
    input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    output = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS
    )

    # Decode and return result
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("assistant\n", 1)[-1]
    answer += f"\n\nЧитайте подробнее по ссылке: {context[0]['source_url']}"
    return answer
