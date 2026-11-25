"""Who writes the replies (write.writer in config.yaml):
    null           the steered model writes its own programs (ModelWriter)
    a model name   another model writes; under mlx-community/ it runs on MLX (MLXWriter)
    files          someone else writes: each prompt is saved as <out>/s<session>_<step>.prompt.txt,
                   and the loop waits for <out>/s<session>_<step>.reply.txt (FileWriter)
"""
import os, time

from . import lang
from ..model.generate import generate
from ..setup.log import log


class ModelWriter:
    """A language model writes the replies, with constrained decoding so every reply is valid text."""

    def __init__(self, m, w):
        self.m, self.w = m, w
        self.grammar = lang.grammar(w["actions"])
        self.vocab = token_pieces(m)
        end = "END" if w["actions"] == 1 else "DONE"
        self.stop = lambda t: t.rstrip().endswith(end)

    def reply(self, conversations, step):
        return generate(self.m, conversations, stop=self.stop, temperature=self.w["temperature"],
                        max_new_tokens=300 * self.w["actions"], grammar=self.grammar, vocab=self.vocab,
                        repetition_penalty=self.w["repetition_penalty"])


class MLXWriter:
    """A quantized model on MLX writes the replies (for writers too big for memory in bf16).
    Same constrained decoding as ModelWriter, one conversation at a time."""

    def __init__(self, path, w):
        import types
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        self.w = w
        self.model, self.wrap = load(path)
        self.tok = self.wrap._tokenizer
        self.rx = __import__("regex").compile(lang.grammar(w["actions"]))
        self.index = {}                                       # candidates grouped by first character
        for tid in token_pieces(types.SimpleNamespace(tok=self.tok)):
            t = self.tok.decode([tid])
            if t:
                self.index.setdefault(t[0], []).append((tid, t))
        self.ends = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")]
        self.sampler = make_sampler(temp=w["temperature"], top_p=0.95)
        self.penalty = make_logits_processors(repetition_penalty=w["repetition_penalty"])
        end = "END" if w["actions"] == 1 else "DONE"
        self.stop = lambda t: t.rstrip().endswith(end)

    def _mask(self, start):
        import mlx.core as mx
        import numpy as np

        def proc(tokens, logits):
            text = self.tok.decode(np.array(tokens[start:]).tolist(), skip_special_tokens=True)
            if self.rx.fullmatch(text):
                ok = self.ends
            else:
                ok = [tid for ch, toks in self.index.items() if self.rx.fullmatch(text + ch, partial=True)
                      for tid, t in toks if self.rx.fullmatch(text + t, partial=True)] or self.ends
            allow = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
            allow[ok] = 0.0
            return logits + mx.array(allow)
        return proc

    def reply(self, conversations, step):
        from mlx_lm import stream_generate
        out = []
        for conv in conversations:
            prompt = self.tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False)
            if not isinstance(prompt, list):
                prompt = prompt["input_ids"]
            text = ""
            for r in stream_generate(self.model, self.wrap, prompt, max_tokens=300 * self.w["actions"],
                                     sampler=self.sampler, logits_processors=self.penalty + [self._mask(len(prompt))]):
                text += r.text
                if self.stop(text):
                    break
            out.append(text)
        return out


class FileWriter:
    """Someone else writes the replies, into files."""

    def __init__(self, out):
        self.out = out

    def reply(self, conversations, step):
        for i, conv in enumerate(conversations):
            open(f"{self.out}/s{i + 1}_{step}.prompt.txt", "w").write(
                "\n\n".join(f"=== {m['role']} ===\n{m['content']}" for m in conv))
        paths = [f"{self.out}/s{i + 1}_{step}.reply.txt" for i in range(len(conversations))]
        log(f"waiting for {', '.join(paths)}")
        while not all(os.path.exists(p) for p in paths):
            time.sleep(2)
        time.sleep(0.5)
        return [open(p).read() for p in paths]


def token_pieces(m):
    """The token ids that programs are made of: tokenize keywords, numbers, every word name and
    typical statements, and add every single character, so constrained writing never reaches a
    dead end. A short candidate list makes constrained writing fast."""
    names = list(lang.NAME_TO_WORD)
    texts = ["ADD\n", "REPLACE 12\n", "DELETE 3\n", "TEST\n", "END\n", "DONE", "else:\n", "elif best == 8:\n",
             "if best == 0:\n", "    "] + [str(x) for x in range(100)] + [f"{x / 100:.2f}" for x in range(-999, 1000)]
    for n in names:
        texts += [f"set({n}, 1.00)\n", f"    set({n}, -2.50)\n", f"clear({n})\n", f"    clear({n})\n",
                  f"if {n} > 0.50:\n", f"best = argmax({n}, {n})\n", f"TEST {n}\nEND", f"0.50 * {n} + {n} - {n}"]
    ids = set()
    for t in texts:
        ids.update(m.tok(t, add_special_tokens=False).input_ids)
    for ch in lang.GRAMMAR_CHARS:
        one = m.tok(ch, add_special_tokens=False).input_ids
        if len(one) == 1:
            ids.add(one[0])
    return sorted(i for i in ids if set(m.tok.decode([i])) <= lang.GRAMMAR_CHARS)
