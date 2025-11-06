
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_packed_sequence

class CustomDataset(Dataset):

    def __init__(self, inputs: list[list[str]], targets: list[list[int]]):
        super().__init__()

        assert len(inputs) == len(targets)

        self.inputs = inputs
        self.targets = targets

        # self.encodings = tokeniser(texts, paddings='max_lenght', truncation=True, 
        #                            max_length=max_len, return_tensors='pt')
        # self.labels = torch.tensor()
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

    
def collate_fn(batch, pad_id):

    inps = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    inps_p = pad_packed_sequence(inps, batch_first=True, padding_value=pad_id)
    tgts_p = pad_packed_sequence(tgts, batch_first=True, padding_value=pad_id)
    mask = (tgts_p != pad_id)

    return inps_p, tgts_p, mask

