# 加载模型和分词器
from unsloth import FastLanguageModel
from local_dataset import LocalJsonDataset
from safetensors.torch import load_model, save_model

max_seq_length = 2048
dtype = None
load_in_4bit = False
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./model/Qwen2-0.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



# 加载和预处理数据集
custom_dataset = LocalJsonDataset(json_file='train_data.json', tokenizer=tokenizer, max_seq_length=max_seq_length)
dataset = custom_dataset.get_dataset()


# 设置训练配置
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        max_steps=2000,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        #save_strategy="no"
    ),
)


# 训练模型
trainer.train()
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")


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
    
