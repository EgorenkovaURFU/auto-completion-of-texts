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
    
    @torch.no_grad()
    def generate(self, seed_tokens, max_len=20, temperature=1.0, pad_id=None, device='cpu'):
        """
        seed_tokens: list[int] — начальная последовательность
        max_len: сколько токенов генерировать
        temperature: плавность распределения (для сэмплирования)
        """
        self.eval()
        generated = list(seed_tokens)
        
        # Преобразуем seed в тензор
        input_seq = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
        
        hidden = None  # начальное состояние LSTM
        
        for _ in range(max_len):
            emb = self.emb(input_seq)                   # [1, L, emb_dim]
            out, hidden = self.rnn(emb, hidden)        # out: [1, L, hidden], hidden: (h,c)
            logits = self.out(out[:, -1, :])           # последний шаг: [1, vocab_size]
            
            # Применяем temperature
            probs = torch.softmax(logits / temperature, dim=-1)  # [1, vocab_size]
            
            # Сэмплируем следующий токен
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Если паддинг — можно остановить
            if pad_id is not None and next_token == pad_id:
                break
                
            # Добавляем к последовательности
            generated.append(next_token)
            
            # Для следующего шага вход = последний токен
            input_seq = torch.tensor([next_token], dtype=torch.long, device=device).unsqueeze(0)
        
        return generated
