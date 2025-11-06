import torch


class RNN(torch.nn.Module):

    def __init__(self, vacab_size: int, emb_dim: int = 128, hidden: int = 256, padding_idx: int = None):
        super().__init__()

        self.emb = torch.nn.Embedding(vacab_size, emb_dim, padding_idx=padding_idx)
        self.rnn = torch.nn.LSTM(emb_dim, hidden, batch_first=True)
        self.out = torch.nn.Linear(hidden, vacab_size)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.rnn(emb)
        logits = self.out(out)
        return logits
