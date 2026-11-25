"""Replies to chats from the model, with constrained decoding (used to write programs)."""
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList


@torch.no_grad()
def generate(m, conversations, stop, temperature=1.0, max_new_tokens=300, grammar=None, vocab=None,
             repetition_penalty=1.0):
    """Reply to several chats at once. stop(text) -> True ends a reply. With grammar (a regular
    expression) every reply stays a prefix of a match (constrained decoding); vocab lists the
    token ids allowed as candidates (a short list makes this fast)."""
    texts = [m.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for c in conversations]
    side, m.tok.padding_side = m.tok.padding_side, "left"
    try:
        enc = m.tok(texts, return_tensors="pt", padding=True).to(m.device)
    finally:
        m.tok.padding_side = side
    start = enc.input_ids.shape[1]
    procs = LogitsProcessorList([_Grammar(m, start, grammar, vocab)] if grammar else [])
    out = m.model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                           repetition_penalty=repetition_penalty,
                           max_new_tokens=max_new_tokens, pad_token_id=m.tok.eos_token_id,
                           logits_processor=procs,
                           stopping_criteria=StoppingCriteriaList([_Stop(m.tok, start, stop)]))
    return m.tok.batch_decode(out[:, start:], skip_special_tokens=True)


class _Grammar(LogitsProcessor):
    """Constrained decoding: only tokens that keep the reply a prefix of a grammar match.
    Once the reply is a full match, only the end of the reply is allowed."""

    def __init__(self, model, start, pattern, vocab):
        import regex
        self.tok, self.start = model.tok, start
        self.rx = regex.compile(pattern)
        key = tuple(vocab)
        cache = model.__dict__.setdefault("_vocab_index", {})
        if key not in cache:                                 # candidates grouped by first character
            idx = {}
            for tid in vocab:
                t = model.tok.decode([tid])
                if t:
                    idx.setdefault(t[0], []).append((tid, t))
            cache[key] = idx
        self.index = cache[key]
        self.ends = {model.tok.eos_token_id, model.tok.convert_tokens_to_ids("<|im_end|>")}

    def __call__(self, input_ids, scores):
        allow = torch.zeros_like(scores, dtype=torch.bool)
        texts = self.tok.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
        for n, text in enumerate(texts):
            ok = []
            if self.rx.fullmatch(text):
                ok = list(self.ends)
            else:
                for ch, toks in self.index.items():
                    if self.rx.fullmatch(text + ch, partial=True):
                        ok += [tid for tid, t in toks if self.rx.fullmatch(text + t, partial=True)]
            allow[n, ok or list(self.ends)] = True
        return scores.masked_fill(~allow, float("-inf"))


class _Stop(StoppingCriteria):
    def __init__(self, tok, start, stop):
        self.tok, self.start, self.stop = tok, start, stop

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tok.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
        return torch.tensor([self.stop(t) for t in texts], device=input_ids.device)
