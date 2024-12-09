import pandas as pd
import torch
import numpy as np

def read_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

class GSM8K_Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file_path, tokenizer, tiny=False):
        self.data = read_jsonl(jsonl_file_path)
        if tiny:
          self.data = self.data[:100]
        self.q_and_a = tokenizer((self.data['question'] + self.data['answer']).to_list(), padding='max_length', return_tensors='pt')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_value = self.q_and_a['input_ids'][idx]
        attention_mask = self.q_and_a['attention_mask'][idx]
        return input_value, attention_mask

class GSM8K_Val_Dataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file_path, tokenizer, tiny=False, use_noop_prompt=False):
        self.use_noop_prompt = use_noop_prompt
        self.tokenizer = tokenizer
        self.data = pd.read_json(jsonl_file_path, lines=True)
        if tiny:
          self.data = self.data[:100]
        self.train_data = pd.read_json('gsm8k/train.jsonl', lines=True)
        if use_noop_prompt:
            with open("noop_engineered_prompt.txt", "r") as f:
                self.noop_prompt = f.read()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.use_noop_prompt:
            prompt_prefix = self.noop_prompt
        else:
            shots1_8_ind = np.random.choice(self.train_data['question'].index, 8, replace=False)
            shots1_8_q = self.train_data['question'][shots1_8_ind].to_list()
            shots1_8_a = self.train_data['answer'][shots1_8_ind].to_list()
            prompt_prefix = """// preamble or system instruction \
                            As an expert problem solver, solve step by step the following mathematical questions.\n\n"""
            questions = ["\n\n// shot-"+str(i)+"\nQ: " + shots1_8_q[i] + "\nA: Let's think step by step. " + shots1_8_a[i] + "\n" for i in range(8)]
            prompt_prefix = "".join(questions)
            prompt_prefix += "\n\n// target question\nQ: "
        prompt_suffix = "\nA: Let's think step by step. "
        question = [prompt_prefix + self.data['question'][idx] + prompt_suffix]
        tok_question = self.tokenizer(question, padding=True, padding_side='left')
        q = tok_question['input_ids'][0]
        q_mask = tok_question['attention_mask'][0]
        
        return self.data['q_id'][idx], q, q_mask, self.data['answer'][idx]