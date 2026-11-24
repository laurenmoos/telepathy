"""The model: one frozen language model, loaded once. It does two things.

1. It runs programs on sudoku boards. The prompt has 27 lines of 9 digits (0 = empty):
   the 9 rows, then the 9 columns, then the 9 boxes, then the target cell, like r4c7=.
   Every row, column and box has its own line, so each has a line end where it can be read.
   One model pass per board state records:
     - the model's state at the end of the cell's row, column and box lines, at each read layer
     - the model's state at '=' after the last layer, before the final normalization
   Reads and writes are then computed from these numbers, with no further model calls:
     read  = gate direction . state at the line end - threshold   (> 0 means "digit is there")
     write = add strength x (typical state size) x (direction of that digit's output row)
             to the '=' state, then normalize and score the nine digits as the model does.
   Adding the push here gives exactly what a hook on the last layer would give.
   Board states are cached by prompt text, so programs that reach the same board share them.

2. It writes programs: generate() replies to a chat, used by write.py.

The weights never change.
"""
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

import sudoku
import sae as sae_lib
from probes import fit_probe, pick_device


def read_accuracy(gate):
    return gate["acc"] if isinstance(gate, dict) else gate[2]


def box_of(r, c):
    return 3 * (r // 3) + c // 3


def lines(grid):
    rows = [grid[r] for r in range(9)]
    cols = [[grid[r][c] for r in range(9)] for c in range(9)]
    boxes = [[grid[3 * (b // 3) + i][3 * (b % 3) + j] for i in range(3) for j in range(3)] for b in range(9)]
    return rows + cols + boxes


def prompt27(grid, r, c):
    return "\n".join("".join(str(x) for x in ln) for ln in lines(grid)) + f"\nr{r + 1}c{c + 1}="


def line_index(unit, r, c):
    return {"ROW": r, "COL": 9 + c, "BOX": 18 + box_of(r, c)}[unit]


class Model:
    def __init__(self, model_path, read_layers, device="auto", dtype=None):
        self.device = pick_device(device)
        if dtype is None:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
        self.read_layers = read_layers
        self.digit_ids = [self.tok(str(d), add_special_tokens=False).input_ids[0] for d in range(1, 10)]
        W = self.model.lm_head.weight[self.digit_ids].detach().float().cpu()
        self.W = W                                           # the model's output rows for 1-9
        w = W - W.mean(0)
        self.write_dir = w / w.norm(dim=1, keepdim=True)     # one push direction per digit
        self.final_norm = self.model.model.norm
        self.cache = {}                                      # prompt text -> recorded state
        self.gates = {}                                      # (unit, digit, layer) -> a read (see read_value)
        self.saes = {}                                       # layer -> sparse autoencoder, when reads use it
        self.line_means = {}                                 # layer -> [27, d] average state at each line end
        self.state_size = None
        self._last = None
        self.model.model.layers[-1].register_forward_hook(self._grab)

    def _grab(self, mod, inp, out):
        self._last = (out[0] if isinstance(out, tuple) else out)[:, -1].detach().float().cpu()

    def _line_ends(self, offsets):
        """Token index of the line break ending each of the 27 lines."""
        ends = []
        for i in range(27):
            ch = i * 10 + 9
            ends.append(next(t for t, (s, e) in enumerate(offsets) if s <= ch < e))
        return ends

    @torch.no_grad()
    def _forward(self, texts, cells=None, batch=16):
        """One model pass per prompt. With cells given, keep only that cell's row, column
        and box line ends (to keep the cache small); otherwise keep all 27."""
        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = self.tok(chunk, return_tensors="pt", return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            hs = self.model.model(**{k: v.to(self.device) for k, v in enc.items()},
                                  output_hidden_states=True).hidden_states
            for n in range(len(chunk)):
                ends = self._line_ends(offs[n].tolist())
                if cells is not None:
                    r, c = cells[i + n]
                    ends = [ends[line_index(u, r, c)] for u in ("ROW", "COL", "BOX")]
                at_ends = torch.stack([hs[L + 1][n, ends].float().cpu() for L in self.read_layers], 1)  # [lines, nL, d]
                out.append((at_ends, self._last[n].clone()))
        return out

    # ------------------------------------------------------------ gates

    @torch.no_grad()
    def line_end_states(self, n_boards, seed, log=print, batch=16):
        """Qwen reads n_boards board prompts. Returns its state at the end of every line, per
        read layer, for training the autoencoders: {layer: [boards, 27, d]} in float16."""
        rng = random.Random(seed)
        boards = [sudoku.make_board(rng, {}) for _ in range(n_boards)]
        texts = [prompt27(b["board"], b["r"], b["c"]) for b in boards]
        out = []
        for i in range(0, len(texts), batch):
            out += [a.to(torch.float16) for a, _ in self._forward(texts[i:i + batch], batch=batch)]
            log(f"  autoencoder boards {min(i + batch, len(texts))}/{len(texts)}")
        X = torch.stack(out)                                  # [boards, 27, nL, d]
        return {L: X[:, :, li] for li, L in enumerate(self.read_layers)}

    def fit_gates(self, n_boards, seed, log=print, kind="direct", features_per_read=4):
        """Fit one read per (unit, digit, layer): "is this digit in this line?", read at the
        line end. Trained on boards the loop never uses.
        kind "direct": a fitted direction on the model's raw state.
        kind "sae": built from the model's own autoencoder features (self.saes must be set)."""
        rng = random.Random(seed)
        boards = [sudoku.make_board(rng, {}) for _ in range(n_boards)]
        texts = [prompt27(b["board"], b["r"], b["c"]) for b in boards]
        X, lasts = [], []
        for i in range(0, len(texts), 16):
            for at_ends, last in self._forward(texts[i:i + 16]):
                X.append(at_ends); lasts.append(last)
            log(f"  gate boards {min(i + 16, len(texts))}/{len(texts)}")
        X = torch.stack(X)                                   # [boards, 27, nL, d]
        self.state_size = torch.stack(lasts).norm(dim=1).mean().item()
        for u, unit in enumerate(("ROW", "COL", "BOX")):
            ls = list(range(9 * u, 9 * u + 9))
            for li, L in enumerate(self.read_layers):
                if kind == "sae":                       # take out "which line this is" (see line_means)
                    feats = (X[:, ls, li] - self.line_means[L][ls]).reshape(-1, X.shape[-1])
                else:
                    feats = X[:, ls, li].reshape(-1, X.shape[-1])
                for d in range(1, 10):
                    y = torch.tensor([d in lines(b["board"])[l] for b in boards for l in ls])
                    if kind == "sae":
                        self.gates[unit, d, L] = sae_lib.fit_read(self.saes[L], feats, y, features_per_read)
                    else:
                        g, thr, acc = fit_probe(feats[y], feats[~y])
                        self.gates[unit, d, L] = (g, thr, acc)
        report = []
        for unit in ("ROW", "COL", "BOX"):
            for L in self.read_layers:
                accs = [read_accuracy(self.gates[unit, d, L]) for d in range(1, 10)]
                report.append(f"read {unit} at layer {L}: mean held-out accuracy {sum(accs) / 9:.2f} "
                              f"(lowest digit {min(accs):.2f})")
        if kind == "sae":
            for unit in ("ROW", "COL", "BOX"):
                L = self.read_layers[0]
                report.append(f"read {unit} at layer {L}, digit 5, uses the model's features "
                              f"{self.gates[unit, 5, L]['ids'].tolist()}")
        return report

    def read_value(self, gate, x, L, memo, line=None):
        """A read's answer for state x at layer L, at line `line` (0-26): > 0 means "the digit is there"."""
        if isinstance(gate, dict):                      # built from autoencoder features
            if L not in memo:
                memo[L] = self.saes[L].encode(x - self.line_means[L][line])
            return float(memo[L][gate["ids"]] @ gate["w"] - gate["thr"])
        g, thr, _ = gate                                # a direction on the raw state
        return float(x @ g - thr)

    def state(self):
        return dict(gates=self.gates, state_size=self.state_size, read_layers=self.read_layers,
                    saes={L: dict(d=s.dec.weight.shape[0], n=s.enc.weight.shape[0], k=s.k, weights=s.state_dict())
                          for L, s in self.saes.items()}, line_means=self.line_means)

    def load_state(self, s):
        self.gates, self.state_size = s["gates"], s["state_size"]
        self.saes, self.line_means = {}, s.get("line_means", {})
        for L, v in s.get("saes", {}).items():
            a = sae_lib.SAE(v["d"], v["n"], v["k"])
            a.load_state_dict(v["weights"])
            self.saes[L] = a.eval()

    # ------------------------------------------------------------ running programs

    def prepare(self, items):
        """Make sure every (grid, r, c) has its model pass cached."""
        need = {}
        for g, r, c in items:
            t = prompt27(g, r, c)
            if t not in self.cache:
                need[t] = (r, c)
        texts = list(need)
        for t, rec in zip(texts, self._forward(texts, [need[t] for t in texts])):
            self.cache[t] = rec

    def answer(self, prog_run, grid, r, c):
        """Run a program (lang.run with a gate reader) on one cell; return the model's digit.
        prepare() must have been called for this board state."""
        at_ends, last = self.cache[prompt27(grid, r, c)]
        li = {L: i for i, L in enumerate(self.read_layers)}

        memos = {u: {} for u in ("ROW", "COL", "BOX")}   # autoencoder features, computed once per line

        def read_gate(unit, d, L):
            x = at_ends[("ROW", "COL", "BOX").index(unit), li[L]]
            return self.read_value(self.gates[unit, d, L], x, L, memos[unit], line_index(unit, r, c))

        h = last.clone()
        for d, s in prog_run(read_gate):
            h += s * self.state_size * self.write_dir[d - 1]
        with torch.no_grad():
            z = self.final_norm(h.to(self.device, self.model.dtype)[None]).float().cpu()[0]
        return int((self.W @ z).argmax()) + 1

    # ------------------------------------------------------------ baseline: the rule in words

    RULE = ("Sudoku rule: every row, every column and every 3x3 box contains each digit from 1 to 9 exactly once.\n"
            "Below are the board's 9 rows, then its 9 columns, then its 9 boxes. 0 means an empty cell.\n"
            "Write the digit that goes in the cell named at the end: the digit missing from that cell's "
            "row, column and box. Answer with one digit only.")

    @torch.no_grad()
    def answer_by_rule(self, items, batch=16):
        """No program. The model is told the rule in words, shown the board, and asked for the
        cell's digit. Its answer is its top choice among the digits 1-9. items: (grid, r, c)."""
        texts = []
        for g, r, c in items:
            board = prompt27(g, r, c)[:-1].rsplit("\n", 1)[0]            # the 27 lines, without the target
            user = f"{self.RULE}\n\n{board}\n\nCell: row {r + 1}, column {c + 1}"
            texts.append(self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False))
        digits = []
        side, self.tok.padding_side = self.tok.padding_side, "left"
        try:
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], return_tensors="pt", padding=True).to(self.device)
                lg = self.model(**enc, logits_to_keep=1).logits[:, -1].float().cpu()
                digits += (lg[:, self.digit_ids].argmax(1) + 1).tolist()
        finally:
            self.tok.padding_side = side
        return digits

    # ------------------------------------------------------------ writing programs

    @torch.no_grad()
    def generate(self, conversations, temperature=0.9, max_new_tokens=120):
        """The same model, used as a chat model: reply to several conversations at once.
        Each reply stops as soon as it contains a complete (seq ...) program."""
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for m in conversations]
        side, self.tok.padding_side = self.tok.padding_side, "left"
        try:
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        finally:
            self.tok.padding_side = side
        start = enc.input_ids.shape[1]
        out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                  max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id,
                                  stopping_criteria=StoppingCriteriaList([_StopAtProgram(self.tok, start)]))
        return self.tok.batch_decode(out[:, start:], skip_special_tokens=True)


def program_closed(text):
    """True once the text holds a (seq ...) whose parentheses have all closed."""
    i = text.lower().find("(seq")
    if i < 0:
        return False
    depth = 0
    for ch in text[i:]:
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            return True
    return False


class _StopAtProgram(StoppingCriteria):
    def __init__(self, tok, start):
        self.tok, self.start = tok, start

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tok.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
        return torch.tensor([program_closed(t) for t in texts], device=input_ids.device)
