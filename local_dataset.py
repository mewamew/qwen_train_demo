import json
from datasets import Dataset


custom_prompt = """下面列出了一个问题. 请写出问题的答案.
### 问题:
{}
### 答案:
{}"""


class LocalJsonDataset:
    def __init__(self, json_file, tokenizer, max_seq_length=2048):
        self.json_file = json_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset = self.load_dataset()

    def load_dataset(self):
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        for item in data:
            text = custom_prompt.format(item['question'], item['answer']) + self.tokenizer.eos_token
            texts.append(text)

        dataset_dict = {
            'text': texts  # 添加'text'字段以适配SFTTrainer
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    def get_dataset(self):
        return self.dataset