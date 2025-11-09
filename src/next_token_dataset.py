
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]["input_ids"]


def make_collate_fn(pad_id: int):
    """
    Возвращает collate_fn, захватывая pad_id из окружения.
    Collate возвращает: inputs_padded [B, L], targets_padded [B, L], mask [B, L], lengths (list[int])
    """
    def collate_fn(batch):
        """
        batch: list of dicts {'input_ids': tensor}
        """
        sequences = [x["input_ids"] if isinstance(x, dict) else x for x in batch]
        sequences = [seq for seq in sequences if seq.numel() > 1]

        if len(sequences) == 0:
            raise ValueError("Все последовательности слишком короткие!")

        inputs = [seq[:-1] for seq in sequences]   # input = все токены кроме последнего
        targets = [seq[1:] for seq in sequences]

        lengths = torch.tensor([t.size(0) for t in inputs], dtype=torch.long)
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_id)
        mask = (targets_padded != pad_id)

        return inputs_padded, targets_padded, mask, lengths

    return collate_fn



# class TextDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_len=128):
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         ids = self.tokenizer.encode(text, truncation=True, max_length=self.max_len)
#         return torch.tensor(ids, dtype=torch.long)
