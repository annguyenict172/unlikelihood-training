import torch
import torch.nn.functional as F

"""
The implementation is largely borrowed from the authors at:
https://github.com/facebookresearch/unlikelihood_training
"""


def token_unlikelihood_loss(logits, targets, padding_idx, alpha=1.0):
    logprobs = F.log_softmax(logits, dim=-1)
    mle_loss = F.nll_loss(logprobs.view(-1, logprobs.size(-1)), targets.reshape(-1), reduction='mean')

    with torch.no_grad():
        ctx_cands = targets.unsqueeze(1).expand(targets.size(0), targets.size(-1), targets.size(-1))
        ctx_cands_ = (ctx_cands.tril(-1) + padding_idx)
        ctx_cands_ = ctx_cands_.triu()
        ctx_cands = ctx_cands.tril(-1) + ctx_cands_

        negative_targets = torch.zeros_like(logprobs).scatter_(-1, ctx_cands, 1)

    one_minus_probs = torch.clamp((1.0 - logprobs.exp()), min=1e-5)
    custom_loss = -torch.log(one_minus_probs) * negative_targets
    custom_loss = custom_loss.mean()

    loss = mle_loss + alpha * custom_loss
    return loss


def generate_completion_greedy_training(model, context, completion_length):
    predicted_tokens = []
    all_logits = []
    past = None
    for i in range(completion_length):
        output, past = model(context, past=past)
        logits = output[:, -1, :]
        token = torch.argmax(logits, dim=1, keepdim=True)
        context = token

        predicted_tokens.append(token)
        all_logits.append(logits)

    predicted_tokens = torch.stack(predicted_tokens, 1)
    predicted_tokens = predicted_tokens.view(predicted_tokens.size(0), predicted_tokens.size(1))

    return predicted_tokens, torch.stack(all_logits, 1)


def sequence_unlikelihood_loss(pred_toks, logits, ngram_size):
    lprobs = F.log_softmax(logits, dim=-1)
    mask = ngram_repeat_mask(pred_toks, ngram_size).type_as(lprobs)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.mean()
    return loss


def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x) - n):
            ng = tuple(xl[j:j + n])
            if ng in seen:
                mask[i, j:j + n] = 1
            seen.add(ng)
    return mask
