"""Shared helpers: choose a device, and fit a gate (a direction plus a threshold)."""
import torch


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def fit_probe(T, F, steps=300):
    """Logistic regression on standardized activations, folded to one direction + threshold.
    Holds out 20% of each class for the reported accuracy."""
    nT, nF = int(len(T) * .8), int(len(F) * .8)
    X = torch.cat([T[:nT], F[:nF]]); y = torch.cat([torch.ones(nT), torch.zeros(nF)])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.01)
    wpos = (len(y) - y.sum()) / y.sum()                       # balance the classes
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            ((X - mu) / sd) @ w + b, y, pos_weight=wpos) + 1e-3 * w.pow(2).sum()
        loss.backward(); opt.step()
    g = (w / sd).detach(); thr = ((mu * g).sum() - b).item()
    acc = (((T[nT:] @ g) > thr).float().mean() + ((F[nF:] @ g) <= thr).float().mean()).item() / 2
    return g, thr, acc
