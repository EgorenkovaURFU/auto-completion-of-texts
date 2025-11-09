from rouge_score import rouge_scorer
import torch

def compute_rouge(model, loader, tokenizer, device, pad_id, gen_fraction=0.25):
    """
    model: обученная LSTM
    loader: DataLoader
    tokenizer: BertTokenizerFast
    gen_fraction: доля текста, которую нужно сгенерировать
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores_list = []

    with torch.no_grad():
        for batch in loader:
            inps_b, tgts_b, mask_b = batch[:3]  # игнорируем лишние элементы
            B, L = inps_b.size()

            inps_b = inps_b.to(device)
            tgts_b = tgts_b.to(device)

            for i in range(B):
                seq = inps_b[i]
                tgt = tgts_b[i]

                # определяем длину seed (3/4)
                L_gen = int(L * gen_fraction)
                L_seed = L - L_gen
                seed = seq[:L_seed].tolist()  # вход для генерации

                # генерируем оставшуюся часть
                gen_seq = model.generate(seed, max_len=L_gen, pad_id=pad_id, device=device)

                # переводим в текст
                gen_text = tokenizer.decode(gen_seq[L_seed:], skip_special_tokens=True)
                tgt_text = tokenizer.decode(tgt[L_seed:].tolist(), skip_special_tokens=True)

                # считаем ROUGE
                scores = scorer.score(tgt_text, gen_text)
                scores_list.append(scores)

    # усредняем по всем примерам
    avg_scores = {}
    for key in ['rouge1', 'rouge2', 'rougeL']:
        avg_scores[key] = sum([s[key].fmeasure for s in scores_list]) / len(scores_list)

    return avg_scores

