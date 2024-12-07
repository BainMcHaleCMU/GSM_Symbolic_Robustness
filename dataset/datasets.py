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
    def __init__(self, jsonl_file_path, tokenizer, tiny=False):
        self.data = pd.read_json(jsonl_file_path, lines=True)
        if tiny:
          self.data = self.data[:100]
        shots1_8_ind = np.random.choice(self.data['question'].index, 8, replace=False)
        shots1_8_q = self.data['question'][shots1_8_ind].to_list()
        shots1_8_a = self.data['answer'][shots1_8_ind].to_list()
        prompt_prefix = """// preamble or system instruction \
                        As an expert problem solver, solve step by step the following mathematical questions.\n\n"""
        questions = ["\n\n// shot-"+str(i)+"\nQ: " + shots1_8_q[i] + "\nA: Let's think step by step. " + shots1_8_a[i] + "\n" for i in range(8)]
        prompt_prefix = "".join(questions)
        prompt_prefix += "\n\n// target quesiton\nQ: "
        prompt_suffix = "\nA: Let's think step by step. "
        self.data['question'] = prompt_prefix + self.data['question'] + prompt_suffix
        self.tok_question = tokenizer(self.data['question'].to_list(), padding=True, padding_side='left')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        q = self.tok_question['input_ids'][idx]
        q_mask = self.tok_question['attention_mask'][idx]
        return self.data['q_id'][idx], q, q_mask, self.data['answer'][idx]