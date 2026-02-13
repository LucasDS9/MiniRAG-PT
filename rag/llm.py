from transformers import AutoTokenizer, AutoModelForCausalLM


def load_llm(model_name="Qwen/Qwen2.5-1.5B-Instruct"):

    print("Carregando LLM...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model


def generate_answer(prompt, tokenizer, model):

    messages = [
        {"role": "system", "content": "Responda sempre em português e seja objetivo."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()
