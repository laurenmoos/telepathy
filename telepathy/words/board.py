"""Board words: directions that show what is in the target cell's row, column and box. No labels.

At the end of each line of the prompt (a row, column or box of 9 digits), Qwen's state at the
line break summarizes that line. A board word is one direction in that state, at one layer.

1. find()      Qwen reads many boards; take its state at the end of the target cell's row,
               column and box line, at each layer in reads.layers, minus its average at that
               line. FastICA gives independent components: directions whose values vary
               independently across boards. They separate on/off facts about the line, which
               principal directions mix. Each is oriented so its rare state is positive.
2. describe()  correlate each word's value with board facts (which digits are in the line, how
               many cells are empty), against a chance level from shuffled boards. The facts only
               describe the words; they never choose them.

    telepathy/run.sh words     build (or load) both kinds of words

Output: cache/<model>_boardwords_....pt (used by setup.settings.load_words) and runs/board_words/REPORT.md.
"""
import os
import torch

from ..model.states import states_at
from ..setup.log import log
from ..sudoku.boards import facts, target_boards
from ..sudoku.prompt import PLACE_WORDS, line_of


def fastica(X, k, iters=500, seed=0):
    """X [boards, d], centered. Returns k unmixing directions in the original space, [d, k]."""
    C = X.T @ X / len(X)
    ev, U = torch.linalg.eigh(C.double())
    idx = ev.argsort(descending=True)[:k]
    E, D = U[:, idx].float(), ev[idx].float()
    Wh = E / D.sqrt()                                     # whitening: z = X @ Wh
    Z = X @ Wh
    W = torch.linalg.qr(torch.randn(k, k, generator=torch.Generator().manual_seed(seed)))[0]
    for _ in range(iters):
        Y = Z @ W.T
        G, Gp = torch.tanh(Y), 1 - torch.tanh(Y) ** 2
        Wn = (G.T @ Z) / len(Z) - Gp.mean(0)[:, None] * W
        u, _, vt = torch.linalg.svd(Wn)
        Wn = u @ vt                                       # symmetric decorrelation
        done = torch.abs((Wn * W).sum(1)).min() > 1 - 1e-6
        W = Wn
        if done:
            break
    return Wh @ W.T


def find(m, bs, rc):
    """The board words: rc["components"] independent components per line end and layer."""
    need = [(p, L) for p in ("ROW", "COL", "BOX") for L in rc["layers"]]
    X, _ = states_at(m, bs, need)
    out = []
    for place, L in need:
        lines = torch.tensor([line_of(place, b["r"], b["c"]) for b in bs])
        x = X[place, L]
        mean = torch.stack([x[lines == i].mean(0) if (lines == i).any() else x.mean(0) for i in range(9)])
        d = x - mean[lines]
        U = fastica(d, rc["components"])
        V = d @ U
        sign = torch.sign((((V - V.mean(0)) / V.std(0)) ** 3).mean(0))      # rare state positive
        for rank in range(U.shape[1]):
            u = U[:, rank] * sign[rank]
            out.append(dict(place=place, layer=L, u=u / u.norm(), mean=mean,
                            sd=(V[:, rank].std().item() + 1e-6) / u.norm().item(), rank=rank))
        log(f"  {place} line end, layer {L}: {U.shape[1]} components")
    return out


def describe(m, ws, bs, top=3, shuffles=50):
    """Each word's strongest board facts, and the chance level (the 95th percentile of the
    largest correlation with the boards shuffled)."""
    F = [facts(b) for b in bs]
    names = list(F[0])
    Fm = torch.tensor([[f[k] for k in names] for f in F])
    Fz = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-6)
    X, _ = states_at(m, bs, sorted({(w["place"], w["layer"]) for w in ws}))
    g = torch.Generator().manual_seed(0)
    for w in ws:
        lines = torch.tensor([line_of(w["place"], b["r"], b["c"]) for b in bs])
        v = (X[w["place"], w["layer"]] - w["mean"][lines]) @ w["u"]
        vz = (v - v.mean()) / (v.std() + 1e-6)
        corr = (vz[:, None] * Fz).mean(0)
        w["chance"] = sorted((vz[torch.randperm(len(vz), generator=g)][:, None] * Fz).mean(0).abs().max().item()
                             for _ in range(shuffles))[int(0.95 * shuffles)]
        w["facts"] = [(names[j], round(corr[j].item(), 2)) for j in corr.abs().argsort(descending=True)[:top].tolist()]
        above = [(n, c) for n, c in w["facts"] if abs(c) > w["chance"]]
        w["description"] = (f"{PLACE_WORDS[w['place']]}, layer {w['layer']}. "
                            + ("Goes up with: " + "; ".join(f"{n} ({c:+.2f})" for n, c in above) if above
                               else "No board fact above chance") + f" (chance level {w['chance']:.2f}).")


def path(cfg):
    name = cfg["model_path"].rstrip("/").split("/")[-1]
    rc = cfg["reads"]
    return (f"cache/{name}_boardwords_{rc['boards']}boards_seed{rc['seed']}_ica{rc['components']}"
            f"_L{'-'.join(map(str, rc['layers']))}_b{rc['blanks'][0]}-{rc['blanks'][1]}.pt")


def build(m, cfg, log=log):
    rc = cfg["reads"]
    log(f"board words: Qwen reads {rc['boards']} boards")
    ws = find(m, target_boards(rc["boards"], rc["seed"], rc["blanks"]), rc)
    log(f"describing them on {rc['describe_boards']} new boards")
    describe(m, ws, target_boards(rc["describe_boards"], rc["seed"] + 1, rc["blanks"]))
    out = dict(reads=ws, config=rc, model=cfg["model_path"])
    os.makedirs("cache", exist_ok=True)
    torch.save(out, path(cfg))
    report(out)
    return out


def load(m, cfg, log=log):
    p = path(cfg)
    if os.path.exists(p):
        log(f"board words loaded from {p}")
        return torch.load(p, weights_only=False)
    return build(m, cfg, log)


def report(out, where="runs/board_words"):
    os.makedirs(where, exist_ok=True)
    rc = out["config"]
    L = ["# Board words", "",
         f"Model {out['model']}. {rc['boards']} boards ({rc['blanks'][0]}-{rc['blanks'][1]} empty cells), "
         f"{rc['components']} independent components per line end and layer.", ""]
    L += [f"{i}. {w['description']}" for i, w in enumerate(out["reads"], 1)]
    open(f"{where}/REPORT.md", "w").write("\n".join(L) + "\n")
    log(f"report: {where}/REPORT.md")

