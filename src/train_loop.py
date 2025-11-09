import torch
from sklearn.metrics import accuracy_score


def train_epoch(model, loader, optimizer, criterion, device):
    """ 
    Функция для обучения нейросети
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0  # для усреднения по токенам (не включая паддинги)

    for inps_b, tgts_b, mask_b, _ in loader:
        inps_b = inps_b.to(device)        # [B, L]
        tgts_b = tgts_b.to(device)        # [B, L]
        mask_b = mask_b.to(device)        # [B, L] bool

        optimizer.zero_grad()
        logits = model(inps_b)            # ожидаем [B, L, V]
        if logits.dim() != 3:
            raise RuntimeError(f"Expected model output [B,L,V], got {logits.shape}")

        B, L, V = logits.size()

        logits_flat = logits.view(B * L, V) 
        targets_flat = tgts_b.view(B * L)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Считаем метрики: суммируем loss (scale по батчам)
        # Для корректного усреднения берём количество ненулевых токенов (mask)
        n_tokens = mask_b.sum().item()
        total_loss += loss.item() * n_tokens           # loss суммирован по токенам
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    return avg_loss


def evaluate(model, loader, criterion, device):
    """ 
    Функция для оценки модели после эпохи обучения
    """

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for inps_b, tgts_b, mask_b, _ in loader:
            inps_b = inps_b.to(device)
            tgts_b = tgts_b.to(device)
            mask_b = mask_b.to(device)

            logits = model(inps_b)               # [B, L, V]
            B, L, V = logits.size()

            logits_flat = logits.view(B * L, V)
            targets_flat = tgts_b.view(B * L)

            loss = criterion(logits_flat, targets_flat)
            n_tokens = mask_b.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            # Предсказания: argmax по измерению словаря -> shape [B, L]
            preds = torch.argmax(logits, dim=2).cpu()   # [B, L]
            trues = tgts_b.cpu()                        # [B, L]

            # Добавляем только непаддинговые позиции в списки для accuracy
            mask_cpu = mask_b.cpu()
            preds_flat = preds.view(-1)
            trues_flat = trues.view(-1)
            mask_flat = mask_cpu.view(-1)
            if mask_flat.sum().item() > 0:
                all_preds.extend(preds_flat[mask_flat].tolist())
                all_trues.extend(trues_flat[mask_flat].tolist())

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    accuracy = accuracy_score(all_trues, all_preds) if len(all_trues) > 0 else float('nan')
    return avg_loss, accuracy
