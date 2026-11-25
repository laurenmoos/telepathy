"""Answer words: directions at the answer position that change Qwen's answer. No labels.

The words are a small action space. An action is a direction u at a place (the answer
position '=' or the end of the target cell's row, column or box line) and a layer, pushed by
one board spread. On board b it changes Qwen's answer distribution by a_b = R_b J_b u * spread:
J_b is the Jacobian of the 9 digit scores, and R_b^T R_b is the Fisher metric of the answer
distribution, so |a_b|^2 is twice the KL divergence of the push, to first order. A set of words
is scored by its empowerment, averaged over boards:
    mean over boards of log det(I + sum_k a_bk a_bk^T / noise)
Per board, a word that controls what an earlier word already controls adds little, so the
variety of boards gives the spread.

1. record()     Qwen reads many boards. At every layer and place: the state x, its covariance,
                and the gradient g of Qwen's answer score.
2. pool()       Candidates at every place and layer: the eigenvectors of
                    M = mean over boards of (g d^T + d g^T) / 2,     d = x - (average x)
                (directions along which board differences move the answer), and the
                principal directions of the states (where boards differ most).
3. jacobians()  On new boards: each candidate's Fisher-weighted effect on the answer, the same
                through the direct path only (straight to the output, skipping later layers),
                and each board's position along it.
4. empower()    Choose words one at a time by the gain in mean log det. At most max_direct
                words may act mostly straight to the answer; a word whose effect over boards
                is nearly parallel to a chosen word of the same kind is skipped.
5. check()      On other boards, change each word and watch the answer (remove, copy from
                another board, push 2 spreads each way), with random directions as controls.
6. describe()   Correlate each word's reading with board facts. The facts only describe the
                words; they are never used to find or choose them.

    python words.py [--set words.boards=64 ...]     build (or load) the words and write the report

Output: word_cache/<model>_primitives_..._logdet.pt (used by core.load_words), the cached Jacobians
next to it (choosing again from them takes seconds), and runs/word_stage_logdet/REPORT.md.
"""
import json, os, random, sys, time
import torch

import sudoku
from model import box_of, prompt27

PLACES = ("ANSWER", "ROW", "COL", "BOX")
PLACE_WORDS = {"ANSWER": "answer position", "ROW": "row line end", "COL": "column line end", "BOX": "box line end"}


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)


# ---------------------------------------------------------------- boards

def boards(n, seed, blanks):
    """Finishable boards with a random empty cell as the target."""
    rng = random.Random(seed)
    lo, hi = blanks
    out = []
    for _ in range(n):
        b = sudoku.make_finishable_board(rng, rng.randint(lo, hi))
        r, c = rng.choice([(r, c) for r in range(9) for c in range(9) if b["board"][r][c] == 0])
        out.append(dict(board=b["board"], solution=b["solution"], r=r, c=c))
    return out


def line_of(place, r, c):
    """Which line of its kind the place belongs to (0 for the answer position)."""
    return {"ANSWER": 0, "ROW": r, "COL": c, "BOX": box_of(r, c)}[place]


def encode(m, bs):
    """Tokens for a batch of boards, and the token index of each place in each board."""
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


# ---------------------------------------------------------------- 1. record

def record(m, bs, batch=8, cov=False):
    """Accumulate, per (place, layer): sum of g x^T, and per line the sums of g and x and the
    count; also the sum of |x|, and with cov the sum of x x^T. Streams over boards, so nothing
    per board is kept."""
    nL, d = len(m.layers), m.d
    A = torch.zeros(len(PLACES), nL, d, d)
    XX = torch.zeros(len(PLACES), nL, d, d) if cov else None
    Sg = torch.zeros(len(PLACES), nL, 9, d)
    Sx = torch.zeros(len(PLACES), nL, 9, d)
    cnt = torch.zeros(len(PLACES), 9)
    size = torch.zeros(len(PLACES), nL)
    for i in range(0, len(bs), batch):
        chunk = bs[i:i + batch]
        enc, pos = encode(m, chunk)
        outs = []
        hooks = [layer.register_forward_hook(lambda mod, inp, out: outs.append(out[0] if isinstance(out, tuple) else out))
                 for layer in m.layers]
        try:
            with torch.enable_grad():
                lg = m.model(**enc, logits_to_keep=1).logits[:, -1][:, m.digit_ids].float()
                s = lg.max(1).values - lg.mean(1)                    # Qwen's own answer score
                grads = torch.autograd.grad(s.sum(), outs)
        finally:
            for h in hooks:
                h.remove()
        rows = torch.arange(len(chunk))[:, None]
        lines = torch.tensor([[line_of(p, b["r"], b["c"]) for p in PLACES] for b in chunk])
        for L in range(nL):
            x = outs[L].detach()[rows, pos.to(outs[L].device)].float().cpu()      # [B, 4, d]
            g = grads[L][rows, pos.to(grads[L].device)].float().cpu()
            A[:, L] += torch.einsum("bpi,bpj->pij", g, x)
            if cov:
                XX[:, L] += torch.einsum("bpi,bpj->pij", x, x)
            size[:, L] += x.norm(dim=-1).sum(0)
            for p in range(len(PLACES)):
                Sg[p, L].index_add_(0, lines[:, p], g[:, p])
                Sx[p, L].index_add_(0, lines[:, p], x[:, p])
        for p in range(len(PLACES)):
            cnt[p] += torch.bincount(lines[:, p], minlength=9).float()
        del outs, grads
        log(f"  recorded {min(i + batch, len(bs))}/{len(bs)} boards")
    return dict(A=A, XX=XX, Sg=Sg, Sx=Sx, cnt=cnt, size=size / len(bs), n=len(bs))


# ---------------------------------------------------------------- 5. check

def states_at(m, bs, need, batch=16):
    """Each board's state at the (place, layer) pairs in need, and Qwen's answer."""
    layers = sorted({L for _, L in need})
    X = {k: [] for k in need}
    answers = []
    for i in range(0, len(bs), batch):
        chunk = bs[i:i + batch]
        enc, pos = encode(m, chunk)
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


def answers_with(m, bs, place, L, deltas, batch=16):
    """Qwen's answers when deltas[b] is added to board b's state at (place, layer)."""
    out = []
    for i in range(0, len(bs), batch):
        chunk = bs[i:i + batch]
        enc, pos = encode(m, chunk)
        adds = [(n, pos[n, PLACES.index(place)].item(), deltas[i + n]) for n in range(len(chunk))]
        out += m.answers_with_adds(enc, {L: adds})
    return out


def check(m, dirs, bs, seed, label="checked"):
    """Run remove, copy and push on each direction (a candidate or a random control) and
    store the results in its "check". The donor boards depend only on seed, so every call
    with the same seed and boards uses the same donors. Returns Qwen's plain accuracy."""
    rng = random.Random(seed)
    need = sorted({(c["place"], c["layer"]) for c in dirs})
    X, base = states_at(m, bs, need)
    right = [b["solution"][b["r"]][b["c"]] for b in bs]
    donor = list(range(len(bs)))
    rng.shuffle(donor)
    differ = [b for b in range(len(bs)) if base[b] != base[donor[b]]]
    lines = {place: torch.tensor([line_of(place, b["r"], b["c"]) for b in bs]) for place in PLACES}

    def run(c, u):
        x = X[c["place"], c["layer"]]
        proj = ((x - c["mean"][lines[c["place"]]]) @ u)                               # [boards]
        sd = proj.std().item() + 1e-6
        res = dict(sd=sd)
        for name, amount in (("remove", -proj), ("copy", proj[donor] - proj),
                             ("push_up", torch.full_like(proj, 2 * sd)), ("push_down", torch.full_like(proj, -2 * sd))):
            ans = answers_with(m, bs, c["place"], c["layer"], [a * u for a in amount.tolist()])
            res[name] = dict(changed=sum(a != b for a, b in zip(ans, base)) / len(bs),
                             right=sum(a == t for a, t in zip(ans, right)) / len(bs))
            if name == "copy":
                res[name]["follows_donor"] = sum(ans[b] == base[donor[b]] for b in differ) / max(1, len(differ))
        return res

    t0 = time.time()
    for i, c in enumerate(dirs):
        c["check"] = run(c, c["u"])
        kind = "random direction at " if c.get("control") else ""
        log(f"  {label} {i + 1}/{len(dirs)}: {kind}{c['place']} layer {c['layer']}: remove changes "
            f"{c['check']['remove']['changed']:.0%}, copy follows donor {c['check']['copy']['follows_donor']:.0%} "
            f"({time.time() - t0:.0f}s)")
    return dict(right=sum(a == t for a, t in zip(base, right)) / len(bs), donors_differ=len(differ))


def controls_for(cands, per_place, seed, have=()):
    """Random unit directions, per_place at each place that has candidates (minus those in
    have), each at the layer of a random candidate there."""
    rng = random.Random(seed + 1)
    torch.manual_seed(seed + 1)
    out = []
    for place in PLACES:
        at = [c for c in cands if c["place"] == place]
        need = per_place - sum(1 for c in have if c["place"] == place)
        for _ in range(max(0, need) if at else 0):
            c = dict(at[rng.randrange(len(at))], control=True)
            u = torch.randn_like(c["u"])
            c["u"] = u / u.norm()
            c.pop("check", None)
            out.append(c)
    return out


# ---------------------------------------------------------------- 2-4. the action space
#
# The words as a small action space. An action is a word u at a place and layer, pushed by one
# board spread (the spread of real boards along u). On board b it changes Qwen's answer
# distribution by a_b = R_b J_b u * spread, where J_b is the Jacobian of the 9 digit scores with
# respect to the state there, and R_b^T R_b is the Fisher metric of the answer distribution
# (so |a_b|^2 is twice the KL divergence the push causes, to first order).
# A set of words is scored by its empowerment, averaged over boards:
#     mean over boards of log det(I + sum_k a_bk a_bk^T / noise)
# This is per board, so a word that controls the same thing on the same boards as a word
# already chosen adds almost nothing; the next word gains most where the others do not reach.
# Log det is submodular, so choosing one word at a time is close to the best set.
# Log det still gains a little from a word parallel to one already chosen (more strength along
# the same direction). Programs get strength from (scale C), so a candidate whose effect over
# all boards has |cosine| above max_overlap with a chosen word of the same kind is skipped.
# A word that acts through Qwen's later layers may still match a direct word's effect: it is
# the composable route to the same control, which programs can build on.

def pool(rec, rc):
    """Candidate actions at every place and layer: the top eigenvectors of M (directions along
    which board differences move the answer) and the top principal directions of the states
    (directions along which boards differ most)."""
    n, out = rec["n"], []
    for p, place in enumerate(PLACES):
        c = rec["cnt"][p].clamp(min=1)
        for L in range(rec["A"].shape[1]):
            mean = rec["Sx"][p, L] / c[:, None]
            M = (rec["A"][p, L] - torch.einsum("li,lj->ij", rec["Sg"][p, L], mean)) / n
            M = ((M + M.T) / 2).double()
            ev, U = torch.linalg.eigh(M)
            total = ev.abs().sum().item() + 1e-12
            for k in ev.abs().argsort(descending=True)[:rc["pool_eigen"]].tolist():
                out.append(dict(place=place, layer=L, u=U[:, k].float(), source="answer eigenvector",
                                eigenvalue=ev[k].item(), share=abs(ev[k].item()) / total, mean=mean))
            C = (rec["XX"][p, L] - torch.einsum("li,lj,l->ij", rec["Sx"][p, L], rec["Sx"][p, L], 1 / c)) / n
            ev, U = torch.linalg.eigh(C.double())
            total = ev.sum().item() + 1e-12
            for k in ev.argsort(descending=True)[:rc["pool_pcs"]].tolist():
                out.append(dict(place=place, layer=L, u=U[:, k].float(), source="board variation",
                                eigenvalue=ev[k].item(), share=ev[k].item() / total, mean=mean))
    return out


def jacobians(m, bs, cands, batch=8):
    """For each board and candidate: the Fisher-weighted change in Qwen's answer per unit push
    (J [boards, 9, candidates]), the same through the direct path only (D: the push carried
    straight to the output, skipping every later layer; zero away from the answer position),
    and the board's position along the candidate (proj [boards, candidates])."""
    nL, last = len(m.layers), len(m.layers) - 1
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault((c["place"], c["layer"]), []).append(i)
    U = {k: torch.stack([cands[i]["u"] for i in idx], 1) for k, idx in groups.items()}
    means = {k: cands[idx[0]]["mean"] for k, idx in groups.items()}
    at_answer = torch.tensor([c["place"] == "ANSWER" for c in cands])
    Uall = torch.stack([c["u"] for c in cands], 1)
    N, P = len(bs), len(cands)
    J, D, proj = torch.zeros(N, 9, P), torch.zeros(N, 9, P), torch.zeros(N, P)
    t0 = time.time()
    for i in range(0, N, batch):
        chunk = bs[i:i + batch]
        B = len(chunk)
        enc, pos = encode(m, chunk)
        lines = {place: torch.tensor([line_of(place, b["r"], b["c"]) for b in chunk]) for place in PLACES}
        outs = []
        hooks = [layer.register_forward_hook(lambda mod, inp, out: outs.append(out[0] if isinstance(out, tuple) else out))
                 for layer in m.layers]
        try:
            with torch.enable_grad():
                lg = m.model(**enc, logits_to_keep=1).logits[:, -1][:, m.digit_ids].float()
                p = lg.softmax(1).detach()
                R = p.sqrt()[:, :, None] * (torch.eye(9, device=p.device) - p[:, None, :])   # R^T R = diag(p) - p p^T
                rows = torch.arange(B)
                for j in range(9):
                    gs = torch.autograd.grad((lg * R[:, j]).sum(), outs, retain_graph=j < 8)
                    for (place, L), idx in groups.items():
                        g = gs[L][rows, pos[:, PLACES.index(place)].to(gs[L].device)].float().cpu()
                        J[i:i + B, j, idx] = g @ U[place, L]
                    gfin = gs[last][rows, pos[:, 0].to(gs[last].device)].float().cpu()
                    D[i:i + B, j] = gfin @ Uall
                    del gs
        finally:
            for h in hooks:
                h.remove()
        for (place, L), idx in groups.items():
            x = outs[L].detach()[rows, pos[:, PLACES.index(place)].to(outs[L].device)].float().cpu()
            proj[i:i + B, idx] = (x - means[place, L][lines[place]]) @ U[place, L]
        del outs
        log(f"  jacobians {min(i + batch, N)}/{N} boards ({time.time() - t0:.0f}s)")
    D[:, :, ~at_answer] = 0
    return J, D, proj


def empower(J, D, proj, cands, rc):
    """Choose words one at a time by the gain in mean log det. At most max_direct words may
    act mostly through the direct path, so most words are ones Qwen's later layers build on."""
    sd = proj.std(0) + 1e-6
    a, ad = J * sd[None, None, :], D * sd[None, None, :]                  # a push of one board spread
    noise = rc["noise"] * (a ** 2).sum(1).mean(0).max()                  # relative to the strongest single word
    a, ad = a / noise ** 0.5, ad / noise ** 0.5
    direct = ((ad ** 2).sum(1).mean(0) /
              ((ad ** 2).sum(1).mean(0) + ((a - ad) ** 2).sum(1).mean(0) + 1e-12))
    for c, s, dr, sz in zip(cands, sd.tolist(), direct.tolist(), (a ** 2).sum(1).mean(0).tolist()):
        c.update(sd=s, directness=dr, alone=sz)
    N = a.shape[0]
    flat = a.permute(2, 0, 1).reshape(a.shape[2], -1)
    flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-12)               # each word's effect over all boards
    Cinv = torch.eye(9).repeat(N, 1, 1)
    chosen, n_direct = [], 0
    for _ in range(rc["primitives"]):
        Ca = torch.einsum("bij,bjp->bip", Cinv, a)                         # [N, 9, P]
        q = (a * Ca).sum(1)                                               # a^T C^-1 a, [N, P]
        gain = torch.log1p(q).mean(0)
        for k in chosen:
            gain[k] = -1
        if n_direct >= rc["max_direct"]:
            gain[direct > 0.5] = -1
        if chosen:                       # a word parallel to a chosen one adds only strength, which (scale C) gives
            same = (direct[:, None] > 0.5) == (direct[chosen][None, :] > 0.5)   # only against words of the same kind
            overlap = ((flat @ flat[chosen].T).abs() * same).max(1).values
            gain[overlap > rc.get("max_overlap", 1.0)] = -1
        k = int(gain.argmax())
        if gain[k] <= 0:
            break
        v = Ca[:, :, k]                                                   # Sherman-Morrison update
        Cinv = Cinv - torch.einsum("bi,bj->bij", v, v) / (1 + q[:, k])[:, None, None]
        chosen.append(k)
        n_direct += int(direct[k] > 0.5)
        cands[k]["gain"] = gain[k].item()
        log(f"  word {len(chosen)}: {cands[k]['place']} layer {cands[k]['layer']} ({cands[k]['source']}), "
            f"gain {gain[k]:.3f}, alone {cands[k]['alone']:.3f}, direct share {direct[k]:.0%}")
    total = sum(cands[k]["gain"] for k in chosen)
    return [cands[k] for k in chosen], total


def build(m, cfg, log=log):
    rc = cfg["words"]
    log(f"word stage, step 1: Qwen reads {rc['boards']} boards; state, covariance and answer gradient "
        f"at {len(PLACES)} places x {len(m.layers)} layers")
    jp = path(cfg).replace(".pt", f"_jac{rc['jac_boards']}.pt")
    cands = None
    if not os.path.exists(jp):
        rec = record(m, boards(rc["boards"], rc["seed"], rc["blanks"]), cov=True)
        cands = pool(rec, rc)
        del rec
    log(f"step 2: candidate actions; Fisher-weighted Jacobians on {rc['jac_boards']} new boards")
    if os.path.exists(jp):
        log(f"  loaded from {jp}")
        J, D, proj, cands = torch.load(jp, weights_only=False)
    else:
        J, D, proj = jacobians(m, boards(rc["jac_boards"], rc["seed"] + 2, rc["blanks"]), cands)
        os.makedirs("word_cache", exist_ok=True)
        torch.save((J, D, proj, cands), jp)
    log("step 3: choosing words by empowerment (mean log det over boards)")
    prims, total = empower(J, D, proj, cands, rc)
    log(f"  mean log det of the chosen set: {total:.3f}")
    for p in prims:
        p["passes"] = True
    controls = controls_for(prims, rc["controls_per_place"], rc["seed"])
    log(f"step 4: checking {len(prims)} words and {len(controls)} random directions on {rc['check_boards']} new boards")
    check_bs = boards(rc["check_boards"], rc["seed"] + 1, rc["blanks"])
    base = check(m, prims, check_bs, rc["seed"])
    check(m, controls, check_bs, rc["seed"], label="control")
    best = {}
    for place in PLACES:
        cs = [c for c in controls if c["place"] == place]
        if cs:
            best[place] = dict(remove=max(c["check"]["remove"]["changed"] for c in cs),
                               copy=max(c["check"]["copy"]["follows_donor"] for c in cs))
    for p in prims:
        b = best[p["place"]]
        p["passes"] = p["check"]["remove"]["changed"] > b["remove"] or p["check"]["copy"]["follows_donor"] > b["copy"]
    log("step 5: describing the words by board facts")
    describe(m, prims, check_bs)
    for p in prims:
        how = "mostly straight to the answer" if p["directness"] > 0.5 else "mostly through Qwen's later layers"
        p["description"] += f". Acts {how} (direct share {p['directness']:.0%})"
    summary = [dict(place=c["place"], layer=c["layer"], source=c["source"], alone=c["alone"],
                    directness=c["directness"], sd=c["sd"]) for c in cands]
    out = dict(primitives=prims, candidates=prims, controls=controls, base=base, controls_best=best,
               logdet=total, pool=summary, config=rc, model=cfg["model_path"])
    os.makedirs("word_cache", exist_ok=True)
    torch.save(out, path(cfg))
    report(out)
    return out


# ---------------------------------------------------------------- 6. describe

def facts(b):
    """Board facts about the target cell, used only to describe primitives."""
    g, r, c = b["board"], b["r"], b["c"]
    units = {"row": sudoku.unit_cells(r, c, "row"), "column": sudoku.unit_cells(r, c, "col"),
             "box": sudoku.unit_cells(r, c, "box")}
    f = {}
    for name, cells in units.items():
        vals = [g[i][j] for i, j in cells]
        for d in range(1, 10):
            f[f"{d} is in the {name}"] = float(d in vals)
        f[f"empty cells in the {name}"] = float(vals.count(0))
    f["candidates for the cell"] = float(len(sudoku.all_candidates(g, r, c)))
    for d in range(1, 10):
        f[f"the answer is {d}"] = float(b["solution"][r][c] == d)
    return f


def describe(m, prims, bs, top=3):
    need = sorted({(p["place"], p["layer"]) for p in prims})
    X, base = states_at(m, bs, need)
    F = [facts(b) for b in bs]
    names = list(F[0])
    Fm = torch.tensor([[f[k] for k in names] for f in F])
    for d in range(1, 10):                                  # also: which digit Qwen itself picks
        names.append(f"Qwen picks {d}")
        Fm = torch.cat([Fm, torch.tensor([[float(a == d)] for a in base])], 1)
    for p in prims:
        lines = torch.tensor([line_of(p["place"], b["r"], b["c"]) for b in bs])
        reading = (X[p["place"], p["layer"]] - p["mean"][lines]) @ p["u"]
        rz = (reading - reading.mean()) / (reading.std() + 1e-6)
        Fz = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-6)
        corr = (rz[:, None] * Fz).mean(0)
        best = corr.abs().argsort(descending=True)[:top].tolist()
        p["facts"] = [(names[k], round(corr[k].item(), 2)) for k in best]
        p["description"] = (f"{PLACE_WORDS[p['place']]}, layer {p['layer']}. Reading goes up with: "
                            + "; ".join(f"{n} ({r:+.2f})" for n, r in p["facts"]))


# ---------------------------------------------------------------- cached

def path(cfg):
    name = cfg["model_path"].rstrip("/").split("/")[-1]
    rc = cfg["words"]
    return f"word_cache/{name}_primitives_{rc['boards']}boards_seed{rc['seed']}_logdet.pt"


def load(m, cfg, log=log):
    p = path(cfg)
    if os.path.exists(p):
        log(f"primitives loaded from {p}")
        return torch.load(p, weights_only=False)
    return build(m, cfg, log)


def report(out, where="runs/word_stage_logdet"):
    os.makedirs(where, exist_ok=True)
    rc, base, worst = out["config"], out["base"], out["controls_best"]
    L = ["# Word stage", "",
         f"Model {out['model']}. {rc['boards']} boards to find directions, {rc['check_boards']} new boards to check them "
         f"({rc['blanks'][0]}-{rc['blanks'][1]} empty cells). With no change, Qwen gets {base['right']:.0%} of the check "
         f"cells right.", "",
         "Random directions at the same place (the bar each candidate must beat):", ""]
    for place, b in worst.items():
        L.append(f"- {PLACE_WORDS[place]}: removing changes at most {b['remove']:.0%} of answers; copying makes at most "
                 f"{b['copy']:.0%} follow the donor.")
    L += ["",
         "## Primitives", ""]
    if not out["primitives"]:
        L.append("No candidate beat every random direction.")
    for i, p in enumerate(out["primitives"], 1):
        ch = p["check"]
        L.append(f"{i}. {p['description']}")
        L.append(f"   - Remove: {ch['remove']['changed']:.0%} of answers change, {ch['remove']['right']:.0%} right. "
                 f"Copy: {ch['copy']['follows_donor']:.0%} follow the donor. Push up / down: "
                 f"{ch['push_up']['changed']:.0%} / {ch['push_down']['changed']:.0%} change, "
                 f"{ch['push_up']['right']:.0%} / {ch['push_down']['right']:.0%} right. "
                 f"Source: {p['source']}. Gain in log det: {p['gain']:.3f}; alone: {p['alone']:.3f}. "
                 f"Beats random: {'yes' if p['passes'] else 'no'}.")
    L += ["", "## All candidates checked", "",
          "| Place | Layer | Direct share | Remove: changed | Copy: follows donor | Push up: right | Push down: right | Passes |",
          "|---|---|---|---|---|---|---|---|"]
    for c in out["candidates"] + out["controls"]:
        ch = c["check"]
        kind = "random" if c.get("control") else ("yes" if c["passes"] else "no")
        d = "-" if c.get("control") else f"{c['directness']:.0%}"
        L.append(f"| {PLACE_WORDS[c['place']]} | {c['layer']} | {d} | {ch['remove']['changed']:.0%} | "
                 f"{ch['copy']['follows_donor']:.0%} | {ch['push_up']['right']:.0%} | {ch['push_down']['right']:.0%} | {kind} |")
    open(f"{where}/REPORT.md", "w").write("\n".join(L) + "\n")
    log(f"report: {where}/REPORT.md")


if __name__ == "__main__":
    import core
    cfg, a = core.settings(sys.argv[1:])
    m = core.load_model(cfg, a)
    load(m, cfg)
