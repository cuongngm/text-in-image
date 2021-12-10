import torch
import torch.nn.functional as F


def subsequent_mask(tgt, padding_symbol):
    tgt_pad_mask = (tgt != padding_symbol).unsqueeze(1).unsqueeze(3)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8)).to(tgt.device)
    tgt_mask = tgt_pad_mask & tgt_sub_mask.bool()
    return tgt_mask

def greedy_decode(model, input, max_len, start_symbol, padding_symbol, device='cpu'):
    batch_size = input.size(0)
    memory = model.encode(input, None)
    
    ys = torch.ones((batch_size, max_len + 2), dtype=torch.long).fill_(padding_symbol).to(device)
    probs = torch.ones((batch_size, max_len + 2), dtype=torch.float).to(device)
    ys[:, 0] = start_symbol
    # early stop mechanics
    check_eos = torch.empty(batch_size, dtype=torch.long)
    check_eos = torch.ones_like(check).to(device)
    for i in range(max_len + 1):
        out = model.decode(memory, None, ys, subsequent_mask(ys, padding_symbol))
        out = model.generator(out)
        prob = F.softmax(out, dim=-1)
        max_probs, next_word = torch.max(prob, dim=-1)
        ys[:, i + 1] = next_word[:, i]
        probs[:, i + 1] = max_probs[:, i]
        if torch.equal(ys[:, i], check_eos):
            break
    return ys, probs
