"""The model (MLX). It knows nothing about tasks, instructions or programs.

    live.view(messages, think)   the chat as the model reads it: text, token ids, the text of each
                                 token, and each token's character span in the text
    live.prefill(ids, hook)      the prompt pass: one forward pass over ids, layer by layer. After
                                 each layer, hook(layer, h) may read the residual stream h [1, T, d]
                                 and return a changed one. Returns the next-token logits.
    live.generate(messages, ...) a reply. The prompt pass runs with the hook, and generation goes on
                                 from its KV cache, so what the hook changed is in every later token.
"""
import numpy as np


class View:
    """The chat as the model reads it."""

    def __init__(self, text, ids, offsets):
        self.text, self.ids, self.offsets = text, list(ids), list(offsets)
        self.tokens = [text[s:e] for s, e in self.offsets]

    def position_of(self, i):
        """The token position holding character i of text."""
        for t, (s, e) in enumerate(self.offsets):
            if s <= i < e:
                return t
        raise ValueError(f"character {i!r} is not in the text")


class Live:
    def __init__(self, path):
        from mlx_lm import load
        self.model, self.wrap = load(path)
        self.tok = self.wrap._tokenizer
        self.layers = self.model.model.layers
        self.stop = {self.tok.eos_token_id, *[self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]}
        self.end_think = self.tok.convert_tokens_to_ids("</think>")

    def view(self, messages, think=True):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=think)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return View(text, enc["input_ids"], enc["offset_mapping"])

    def prefill(self, ids, hook=None, cache=None):
        """The prompt pass. Returns the logits [1, vocab] for the next token."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        h = self.model.model.embed_tokens(mx.array([ids]))
        mask = create_attention_mask(h, cache[0] if cache else None)
        for L, layer in enumerate(self.layers):
            h = layer(h, mask, cache[L] if cache else None)
            if hook is not None:
                h = hook(L, h)
        return self.model.lm_head(self.model.model.norm(h[:, -1:]))[:, -1]

    def logprobs(self, logits):
        import mlx.core as mx
        x = logits.astype(mx.float32)
        return np.array(x - mx.logsumexp(x, axis=-1, keepdims=True))[0]

    def generate(self, messages, max_tokens=768, temperature=0.6, think=True, think_tokens=None,
                 repetition_penalty=None, hook=None):
        """A reply to the chat. With think and think_tokens, the thinking is closed after
        think_tokens tokens and the answer follows, up to max_tokens more."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
        v = self.view(messages, think)
        cache = make_prompt_cache(self.model)
        logits = self.prefill(v.ids, hook, cache)
        sample = make_sampler(temp=temperature, top_p=0.95, top_k=20)
        penalty = make_repetition_penalty(repetition_penalty) if repetition_penalty else None
        history, out = list(v.ids), []

        def feed(ids):
            return self.model(mx.array([ids]), cache=cache)[:, -1]

        def run(logits, n, stop_at_end_think):
            for _ in range(n):
                if penalty:
                    logits = penalty(mx.array(history), logits)
                y = sample(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item()
                history.append(y)
                out.append(y)
                if y in self.stop or (stop_at_end_think and y == self.end_think):
                    return logits, y
                logits = feed([y])
            return logits, None

        if think and think_tokens:
            logits, y = run(logits, think_tokens, True)
            if y in self.stop:
                return self.tok.decode(out)
            if y is None:                          # the budget ran out: close the thinking
                close = self.tok.encode("\n</think>\n\n", add_special_tokens=False)
                history.extend(close)
                out.extend(close)
                logits = feed(close)
            else:
                logits = feed([y])
        run(logits, max_tokens, False)
        return self.tok.decode([t for t in out if t not in self.stop])
