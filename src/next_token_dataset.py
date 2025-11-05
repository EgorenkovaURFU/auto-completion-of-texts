
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, texts, labels, tokeniser, max_len=256):
        super().__init__()
        self.encodings = tokeniser(texts, paddings='max_lenght', truncation=True, 
                                   max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attaention_mask': self.encodings['attantion_mask'][idx],
            'label': self.labels[idx]
        }
    
