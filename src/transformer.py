from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilgpt2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# GPT2 не имеет padding token по умолчанию, добавим
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def generate_completion(model, tokenizer, input_ids, gen_fraction=0.25, max_len=None):
    """
    input_ids: torch.Tensor [L]
    """
    L = input_ids.size(0)
    L_gen = int(L * gen_fraction)
    seed = input_ids[:L - L_gen].unsqueeze(0).to(device)  # batch 1

    gen_output = model.generate(
        seed,
        max_length=(seed.size(1) + L_gen),
        do_sample=False,       # greedy
        pad_token_id=tokenizer.pad_token_id
    )
    return gen_output.squeeze(0)