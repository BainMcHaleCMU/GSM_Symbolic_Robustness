import pandas as pd
import torch
import transformers

def read_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

class GSM8K_Dataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file_path, tokenizer):
        self.data = read_jsonl(jsonl_file_path)
        self.data['question'] = self.data['question'].apply(tokenizer.tokenize)
        self.data['answer'] = self.data['answer'].apply(tokenizer.tokenize)
        longest_question = max(self.data['question'].apply(len))
        longest_answer = max(self.data['answer'].apply(len))
        self.max_len = longest_question + longest_answer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        question = self.data['question'][idx]
        answer = self.data['answer'][idx]
        # add 0 padding
        pad_size = self.max_len - len(question) - len(answer)
        padding = [0] * pad_size
        q_and_a = question + answer + padding
        return (q_and_a, pad_size)