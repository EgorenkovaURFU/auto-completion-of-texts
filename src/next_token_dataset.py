
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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

    
# def collate_fn(batch):

#     # # inps = [b[0] for b in batch]
#     # # tgts = [b[1] for b in batch]
#     # # inps_p = pad_packed_sequence(inps, batch_first=True, padding_value=0)
#     # # tgts_p = pad_packed_sequence(tgts, batch_first=True, padding_value=0)
#     # # mask = (tgts_p != 0)
#     # # return inps_p, tgts_p, mask
#     # inputs = [torch.tensor(x[0], dtype=torch.long) for x in batch]
#     # targets = [torch.tensor(x[1], dtype=torch.long) for x in batch]
#     # masks = [torch.tensor(x[2], dtype=torch.long) for x in batch]
#     # # Паддим по значению pad_id
#     # inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
#     # targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
#     # masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
#     # return inputs_padded, targets_padded, masks_padded

def collate_fn(batch):
    inputs = [torch.tensor(x[0], dtype=torch.long) for x in batch]
    targets = [torch.tensor(x[1], dtype=torch.long) for x in batch]
    
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    mask = (targets_padded != 0)  # тензор bool: где не пад
    return inputs_padded, targets_padded, mask

