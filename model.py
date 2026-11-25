"""The model: one frozen language model. It does two things.

1. It runs programs while it fills in a sudoku cell. The prompt has 27 lines of 9 digits
   (0 = empty): the 9 rows, then the 9 columns, then the 9 boxes, then the target cell, like
   r4c7=. Words (lang.py) sit at the end of the target cell's row, column and box line, or at
   the answer position '=':
       read   = (state there - its average on real boards) . direction / spread over boards
       write  = push the state along the direction until the read gives the target value
   run() executes a program inside one forward pass: at each layer, after Qwen's own layer, it
   runs the program's instructions for that layer, so later layers see the writes.

2. It writes programs: generate() replies to chats, with constrained decoding.

The weights never change.
"""
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList,
                          LogitsProcessor, LogitsProcessorList)


def pick_device(name="auto"):
    """The Mac's GPU if there is one, else CUDA, else the CPU."""
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def box_of(r, c):
    return 3 * (r // 3) + c // 3


def lines(grid):
    rows = [grid[r] for r in range(9)]
    cols = [[grid[r][c] for r in range(9)] for c in range(9)]
    boxes = [[grid[3 * (b // 3) + i][3 * (b % 3) + j] for i in range(3) for j in range(3)] for b in range(9)]
    return rows + cols + boxes


def prompt27(grid, r, c):
    return "\n".join("".join(str(x) for x in ln) for ln in lines(grid)) + f"\nr{r + 1}c{c + 1}="


def line_of(place, r, c):
    """Which line of its kind a place belongs to (0 for the answer position)."""
    return {"ANSWER": 0, "ROW": r, "COL": c, "BOX": box_of(r, c)}[place]


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

    def _encode(self, items):
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

    def _value(self, p, state, r, c):
        """A word's value from Qwen's state at the word's position."""
        mean = p["mean"][line_of(p["place"], r, c)].to(state.device, torch.float32)
        return float((state.float() - mean) @ p["u"].to(state.device, torch.float32)) / p["sd"]

    # ------------------------------------------------------------ running programs

    @torch.no_grad()
    def prepare(self, items, batch=16):
        """Record Qwen's plain answer (its 9 digit scores) for every (grid, r, c) not seen yet."""
        todo = {prompt27(g, r, c): (g, r, c) for g, r, c in items}
        todo = [v for t, v in todo.items() if t not in self.cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            enc, _ = self._encode(chunk)
            lg = self.model(**enc, logits_to_keep=1).logits[:, -1][:, self.digit_ids].float().cpu()
            for n, (g, r, c) in enumerate(chunk):
                self.cache[prompt27(g, r, c)] = lg[n]

    @torch.no_grad()
    def run(self, items, batch=16):
        """Run programs inside Qwen's forward pass. items: (grid, r, c, program).
        Returns (digit, trace, scores) per item: trace lists what each instruction did; scores
        are the 9 digit scores. A program that changes nothing gets Qwen's plain answer
        (prepare() first). If and argmax are decided per board: an instruction runs only where
        its branch was taken."""
        import lang
        out = [None] * len(items)
        todo = []
        for j, (g, r, c, prog) in enumerate(items):
            steps, skipped = lang.schedule(prog)
            note = [f"skipped {op} {lang.WORD_TO_NAME.get(w, w)} (its layer had already run)" for op, w in skipped]
            if any(ins[0] in ("store", "clear") for lst in steps.values() for ins in lst):
                todo.append((j, steps, note))
            else:
                plain = self.cache[prompt27(g, r, c)]
                out[j] = (int(plain.argmax()) + 1, note + ["nothing is written, so Qwen is unchanged"], plain)
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            enc, pos = self._encode([items[j][:3] for j, _, _ in chunk])
            acc = [0.0] * len(chunk)
            decided = [{} for _ in chunk]                # if / argmax id -> the branch this board took
            notes = [{} for _ in chunk]                  # argmax id -> {candidate: (word, value)}
            trace = [list(note) for _, _, note in chunk]
            layers = sorted({L for _, steps, _ in chunk for L in steps})

            def at_layer(L):
                def hook(mod, inp, o):
                    t = (o[0] if isinstance(o, tuple) else o).clone()
                    for n, (j, steps, _) in enumerate(chunk):
                        g, r, c, _ = items[j]
                        value = lambda w: self._value(self.word(w), t[n, pos[n][self.word(w)["place"]]], r, c)
                        name = lambda w: lang.WORD_TO_NAME.get(w, w)
                        for ins in steps.get(L, []):
                            if any(decided[n].get(d) != b for d, b in ins[2]):
                                continue                                  # in a branch this board did not take
                            op = ins[0]
                            if op == "scale":
                                acc[n] *= ins[1]
                            elif op == "const":
                                acc[n] = ins[1]
                            elif op == "if":
                                decided[n][ins[3]] = 0 if acc[n] > ins[1] else 1
                                trace[n].append(f"if: {acc[n]:+.2f} > {ins[1]:.2f} is "
                                                + ("true" if decided[n][ins[3]] == 0 else "false"))
                            elif op == "note":
                                notes[n].setdefault(ins[3][0], {})[ins[3][1]] = (ins[1], value(ins[1]))
                            elif op == "switch":
                                got = notes[n].get(ins[3], {})
                                best = max(got, key=lambda i: got[i][1])
                                decided[n][ins[3]] = best
                                trace[n].append("argmax: " + ", ".join(f"{name(w)} {v:+.2f}" for _, (w, v) in sorted(got.items()))
                                                + f"; best == {best}")
                            elif op == "load":
                                acc[n] = value(ins[1])
                            elif op == "add":
                                acc[n] += value(ins[1])
                            else:                                         # store, clear
                                p = self.word(ins[1])
                                v = value(ins[1])
                                target = acc[n] if op == "store" else 0.0
                                t[n, pos[n][p["place"]]] += ((target - v) * p["sd"] * p["u"].to(t.device, torch.float32)).to(t.dtype)
                                trace[n].append(f"{'set' if op == 'store' else 'clear'} {name(ins[1])}: {v:+.2f} -> {target:+.2f}")
                    return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
                return hook
            hooks = [self.layers[L].register_forward_hook(at_layer(L)) for L in layers]
            try:
                lg = self.model(**enc, logits_to_keep=1).logits[:, -1][:, self.digit_ids].float().cpu()
            finally:
                for h in hooks:
                    h.remove()
            for n, (j, _, _) in enumerate(chunk):
                out[j] = (int(lg[n].argmax()) + 1, trace[n], lg[n])
        return out

    @torch.no_grad()
    def word_values(self, items, batch=16):
        """Every word's value on each (grid, r, c), with no program: a tensor [items, words]."""
        out = []
        layers = sorted({p["layer"] for p in self.words})
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            enc, pos = self._encode(chunk)
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
                out.append([self._value(p, got[p["layer"]][n, pos[n][p["place"]]], r, c) for p in self.words])
        return torch.tensor(out)

    @torch.no_grad()
    def answers_with_adds(self, enc, adds):
        """Qwen's digit for each prompt in enc, with each vector in adds {layer: [(row, position,
        vector)]} added to that layer's output at that row and position (used by words.py)."""
        hooks = [self.layers[L].register_forward_hook(_add_at(lst, self.device, self.model.dtype))
                 for L, lst in adds.items()]
        try:
            lg = self.model(**enc, logits_to_keep=1).logits[:, -1][:, self.digit_ids].float().cpu()
        finally:
            for h in hooks:
                h.remove()
        return (lg.argmax(1) + 1).tolist()

    # ------------------------------------------------------------ writing programs

    @torch.no_grad()
    def generate(self, conversations, stop, temperature=1.0, max_new_tokens=300, grammar=None, vocab=None,
                 repetition_penalty=1.0):
        """Reply to several chats at once. stop(text) -> True ends a reply. With grammar (a regular
        expression) every reply stays a prefix of a match (constrained decoding); vocab lists the
        token ids allowed as candidates (a short list makes this fast)."""
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for m in conversations]
        side, self.tok.padding_side = self.tok.padding_side, "left"
        try:
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        finally:
            self.tok.padding_side = side
        start = enc.input_ids.shape[1]
        procs = LogitsProcessorList([_Grammar(self, start, grammar, vocab)] if grammar else [])
        out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                  repetition_penalty=repetition_penalty,
                                  max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id,
                                  logits_processor=procs,
                                  stopping_criteria=StoppingCriteriaList([_Stop(self.tok, start, stop)]))
        return self.tok.batch_decode(out[:, start:], skip_special_tokens=True)


def _add_at(adds, device, dtype):
    """A forward hook that adds each vector at its (row, position) in a layer's output."""
    adds = [(n, p, v.to(device, dtype)) for n, p, v in adds]

    def hook(mod, inp, out):
        t = (out[0] if isinstance(out, tuple) else out).clone()
        for n, p, v in adds:
            t[n, p] += v
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
    return hook


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
