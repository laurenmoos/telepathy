"""Qwen's state at the places words sit, for many boards at once (used to find words)."""
import torch

from ..sudoku.prompt import PLACES, box_of, prompt27


def encode_boards(m, bs):
    """Tokens for a batch of boards, and the token index of each place in each board, in the
    order of PLACES."""
    texts = [prompt27(b["board"], b["r"], b["c"]) for b in bs]
    enc = m.tok(texts, return_tensors="pt", return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    assert bool(enc["attention_mask"].all()), "board prompts must all have the same length"
    pos = []
    for b, o in zip(bs, offs):
        ends = m._line_ends(o.tolist())
        r, c = b["r"], b["c"]
        pos.append([len(o) - 1, ends[r], ends[9 + c], ends[18 + box_of(r, c)]])
    return {k: v.to(m.device) for k, v in enc.items()}, torch.tensor(pos)


def states_at(m, bs, need, batch=16):
    """Each board's state at the (place, layer) pairs in need, and Qwen's answer."""
    layers = sorted({L for _, L in need})
    X = {k: [] for k in need}
    answers = []
    for i in range(0, len(bs), batch):
        chunk = bs[i:i + batch]
        enc, pos = encode_boards(m, chunk)
        got = {}
        hooks = [m.layers[L].register_forward_hook(
            (lambda L: lambda mod, inp, out: got.__setitem__(L, (out[0] if isinstance(out, tuple) else out).detach()))(L))
            for L in layers]
        try:
            with torch.no_grad():
                lg = m.model(**enc, logits_to_keep=1).logits[:, -1][:, m.digit_ids].float().cpu()
        finally:
            for h in hooks:
                h.remove()
        rows = torch.arange(len(chunk))
        for place, L in need:
            X[place, L].append(got[L][rows, pos[:, PLACES.index(place)].to(got[L].device)].float().cpu())
        answers += (lg.argmax(1) + 1).tolist()
    return {k: torch.cat(v) for k, v in X.items()}, answers
