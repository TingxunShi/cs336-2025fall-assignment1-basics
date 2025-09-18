import torch


def cross_entropy_loss(logits, labels):
    max_value = torch.max(logits, dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_value), dim=-1, keepdim=True))
    logits_correct = torch.gather(logits - max_value, dim=-1, index=labels.unsqueeze(-1))
    return torch.mean(log_sum_exp - logits_correct)
