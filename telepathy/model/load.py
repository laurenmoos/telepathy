"""The model: one frozen language model, and its state at the places words sit.

The prompt is the board (sudoku/prompt.py). Words (programs/lang.py) sit at the end of the
target cell's row, column and box line, or at the answer position '=':
    read   = (state there - its average on real boards) . direction / spread over boards
    write  = push the state along the direction until the read gives the target value
Running programs inside the forward pass is in programs/run.py; writing replies is in
model/generate.py.

The weights never change.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..sudoku.prompt import box_of, line_of, prompt27


def pick_device(name="auto"):
    """The Mac's GPU if there is one, else CUDA, else the CPU."""
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Model:
    def __init__(self, model_path, device="auto", dtype=None):
        self.device = pick_device(device)
        if dtype is None:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
        self.layers = self.model.model.layers
        self.d = self.model.config.hidden_size
        self.digit_ids = [self.tok(str(d), add_special_tokens=False).input_ids[0] for d in range(1, 10)]
        self.cache = {}                                      # prompt text -> Qwen's plain digit scores
        self.words = []                                      # set by set_words()

    def set_words(self, words):
        self.words = words
        self.cache = {}

    def word(self, wid):
        """The word with interpreter id "wK"."""
        return self.words[int(wid[1:]) - 1]

    def _line_ends(self, offsets):
        """Token index of the line break ending each of the 27 lines."""
        return [next(t for t, (s, e) in enumerate(offsets) if s <= i * 10 + 9 < e) for i in range(27)]

    def encode(self, items):
        """Tokens for (grid, r, c) items, and each item's token index for every place."""
        texts = [prompt27(g, r, c) for g, r, c in items]
        enc = self.tok(texts, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        assert bool(enc["attention_mask"].all()), "board prompts must all have the same length"
        pos = []
        for (g, r, c), o in zip(items, offs):
            ends = self._line_ends(o.tolist())
            pos.append({"ANSWER": len(o) - 1, "ROW": ends[r], "COL": ends[9 + c], "BOX": ends[18 + box_of(r, c)]})
        return {k: v.to(self.device) for k, v in enc.items()}, pos

    def value(self, p, state, r, c):
        """A word's value from Qwen's state at the word's position."""
        mean = p["mean"][line_of(p["place"], r, c)].to(state.device, torch.float32)
        return float((state.float() - mean) @ p["u"].to(state.device, torch.float32)) / p["sd"]

    @torch.no_grad()
    def prepare(self, items, batch=16):
        """Record Qwen's plain answer (its 9 digit scores) for every (grid, r, c) not seen yet."""
        todo = {prompt27(g, r, c): (g, r, c) for g, r, c in items}
        todo = [v for t, v in todo.items() if t not in self.cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            enc, _ = self.encode(chunk)
            lg = self.model(**enc, logits_to_keep=1).logits[:, -1][:, self.digit_ids].float().cpu()
            for n, (g, r, c) in enumerate(chunk):
                self.cache[prompt27(g, r, c)] = lg[n]

    @torch.no_grad()
    def word_values(self, items, batch=16):
        """Every word's value on each (grid, r, c), with no program: a tensor [items, words]."""
        out = []
        layers = sorted({p["layer"] for p in self.words})
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            enc, pos = self.encode(chunk)
            got = {}
            hooks = [self.layers[L].register_forward_hook(
                (lambda L: lambda mod, inp, o: got.__setitem__(L, (o[0] if isinstance(o, tuple) else o)))(L))
                for L in layers]
            try:
                self.model(**enc, logits_to_keep=1)
            finally:
                for h in hooks:
                    h.remove()
            for n, (g, r, c) in enumerate(chunk):
                out.append([self.value(p, got[p["layer"]][n, pos[n][p["place"]]], r, c) for p in self.words])
        return torch.tensor(out)

    @torch.no_grad()
    def answers_with_adds(self, enc, adds):
        """Qwen's digit for each prompt in enc, with each vector in adds {layer: [(row, position,
        vector)]} added to that layer's output at that row and position (used by words/answer.py)."""
        hooks = [self.layers[L].register_forward_hook(_add_at(lst, self.device, self.model.dtype))
                 for L, lst in adds.items()]
        try:
            lg = self.model(**enc, logits_to_keep=1).logits[:, -1][:, self.digit_ids].float().cpu()
        finally:
            for h in hooks:
                h.remove()
        return (lg.argmax(1) + 1).tolist()


def _add_at(adds, device, dtype):
    """A forward hook that adds each vector at its (row, position) in a layer's output."""
    adds = [(n, p, v.to(device, dtype)) for n, p, v in adds]

    def hook(mod, inp, out):
        t = (out[0] if isinstance(out, tuple) else out).clone()
        for n, p, v in adds:
            t[n, p] += v
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
    return hook
