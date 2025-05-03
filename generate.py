import torch

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long)

    print(prompt, end="", flush=True)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids[:, -model.config.block_size:])
            logits = logits[:, -1, :] / temperature

            if top_k:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            new_token = tokenizer.decode(next_token[0].tolist())
            print(new_token, end="", flush=True)

    print()
