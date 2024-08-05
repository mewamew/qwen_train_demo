from unsloth import FastLanguageModel


max_seq_length = 2048
dtype = None
load_in_4bit = False
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


FastLanguageModel.for_inference(model)

def generate_answer(question):
    input_text = f"下面列出了一个问题. 请写出问题的答案.\n####问题:{question}\n####答案:"
    inputs = tokenizer(
        [input_text], 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded_output.split('<|im_end|>')[0].strip()

print("请输入您的问题,输入'exit'退出:")
while True:
    user_input = input("> ")
    if user_input.lower() == 'exit':
        print("程序已退出。")
        break
    answer = generate_answer(user_input)
    print("---")
    print(answer)