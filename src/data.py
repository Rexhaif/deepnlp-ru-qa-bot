from tokenizers import Tokenizer
import torch.utils.data as torchdata
import torch
from typing import *


class QAGenDataset(torchdata.Dataset):
    
    def __init__(self, texts: List[str], tokenizer: Tokenizer, max_length: int = 128):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        
        self.tokenizer.enable_padding(pad_token="<PAD>", length=max_length)
        self.tokenizer.enable_truncation(max_length=max_length)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        example = self.texts[idx]
        parts = example.split("\t")
        if len(parts) == 3:
            question, category, response = parts
            encodings = self.tokenizer.encode(f"<START> {question} <CAT> {category} <RESP> {response} <END>")
            input_ids = torch.tensor(encodings.ids, dtype=torch.long)
            mask = torch.tensor(encodings.attention_mask, dtype=torch.bool)
            return input_ids, mask
        elif len(parts) == 2:
            question, response = parts
            encodings = self.tokenizer.encode(f"<START> {question} <RESP> {response} <END>")
            input_ids = torch.tensor(encodings.ids, dtype=torch.long)
            mask = torch.tensor(encodings.attention_mask, dtype=torch.bool)
            return input_ids, mask
        else:
            raise ValueError("Incorrect Example found, no tabs")
