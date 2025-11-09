import optuna
import torch
from torch.utils.data import DataLoader
from src.lstm_model import RNN



def make_objective(tokenizer, pad_id, collate, train_ds, val_ds, train_epoch, criterion, evaluate, num_epochs, device='cpu'):

    def objective(trial):
        """ 
        Функция для подбора гиперпараметров для LSTM модели
        """
        hidden = trial.suggest_categorical('hidden', [128, 256])
        emb = trial.suggest_categorical('emb', [64, 128])
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        bs = trial.suggest_categorical('batch_size', [64, 128])

        print(f'hidden - {hidden}, emb - {emb}, lr - {lr}, bs - {bs}')

        model = RNN(vacab_size=tokenizer.vocab_size, emb_dim=emb, hidden=hidden, padding_idx=pad_id).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=bs, collate_fn=collate)

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}  TrainLoss={train_loss:.6f}  ValLoss={val_loss:.6f} ")
        
        return val_loss
    
    return objective

