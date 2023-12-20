"""Custom text dataset class"""

import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, DistilBertTokenizer

class CustomNewsDataset(torch.utils.data.Dataset):
    """Custom news dataset class"""
    archs = {
        "roberta": (RobertaTokenizer, "roberta-base"),
        "distilbert": (DistilBertTokenizer, "distilbert-base-uncased"),
    }
    def __init__(self, df: pd.DataFrame, arch: str = "distilbert", max_sequence_length: int = 128, lower_case: bool =True):
        """Initialization"""
        tokenizer, version = self.archs[arch]
        self.tokenizer = tokenizer.from_pretrained(version, do_lower_case=lower_case)
        self.df = df
        self.max_sequence_length = max_sequence_length
        
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.df)

    def __getitem__(self, idx):
        """Returns data instance with given index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        label = row["target"]
        heading_and_body = row['title']+' '+row['text']
        encoded = self.tokenizer.encode_plus(
                        heading_and_body,                      
                        add_special_tokens = True,
                        max_length = self.max_sequence_length,
                        padding = 'max_length',
                        return_attention_mask = True,
                        truncation=True,
                        return_tensors='pt',
                        
              )
        encoded = {k:v.squeeze(0) for k,v in encoded.items()}
        encoded['target'] = torch.tensor(label).float()

        return encoded

    def decode(self, text):
        """Returns the decoded text."""
        return self.tokenizer.decode(text, skip_special_tokens=True)
