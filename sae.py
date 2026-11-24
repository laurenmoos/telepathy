"""Reads built from the model's own features.

A sparse autoencoder learns, with no labels, a dictionary of features from the model's
internal state at one layer while Qwen reads many board prompts. It learns from the state
at the end of each line, after taking out the average state for that line position, so
what's left is mostly what is on the line. Each state is rewritten as a few active
features (the top k). Like the Golden Gate feature, each one
fires for something the model itself picked out.

A read ("is digit D in this line?") is then built from those features. A supervised step
picks the few features whose activity best separates lines that contain D from lines that
don't, and fits weights and a threshold on just those. The read does the same job as before,
but it can only use features the model already has.
"""
import math
import torch
import torch.nn as nn



class SAE(nn.Module):
    """Top-k sparse autoencoder: d -> n features, k active per token."""

    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("scale", torch.ones(d))
        self.enc = nn.Linear(d, n)
        self.dec = nn.Linear(n, d)
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.T)
            self.normalize_decoder()
            self.enc.bias.zero_()
            self.dec.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        self.dec.weight.div_(self.dec.weight.norm(dim=0, keepdim=True) + 1e-8)

    def encode(self, x):
        """Feature activity for a state (or a batch of states)."""
        x = (x - self.mu) / self.scale
        pre = torch.relu(self.enc(x - self.dec.bias))
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, top.indices, top.values)

    def forward(self, x):
        f = self.encode(x)
        return self.dec(f), f


def train_sae(X, n, k, steps, batch=1024, lr=1e-3, device="cpu", log=print, standardize=True):
    """Train on states X [tokens, d] (every token Qwen read). Returns the autoencoder and how
    well it rebuilds held-out states (share of variance explained) and how many features never fire.
    standardize: scale each coordinate of the state to unit spread first, so quiet directions
    count as much as loud ones (otherwise the autoencoder spends itself on which token it is)."""
    X = X.float()
    norms = X.norm(dim=1)
    keep = norms < 5 * norms.median()          # drop the few "attention sink" tokens with huge states
    dropped = (~keep).float().mean().item()
    X = X[keep]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g)
    hold, X = X[perm[:len(X) // 10]], X[perm[len(X) // 10:]]
    d = X.shape[1]
    sae = SAE(d, n, k)
    with torch.no_grad():
        sae.mu.copy_(X.mean(0))
        if standardize:
            sae.scale.copy_((X - sae.mu).std(0) + 1e-6)
        else:
            sae.scale.fill_((X - sae.mu).norm(dim=1).mean().item() / math.sqrt(d))
    sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    fired = torch.zeros(n, device=device)
    for step in range(steps):
        x = X[torch.randint(len(X), (batch,), generator=g)].to(device)
        xn = (x - sae.mu) / sae.scale
        xhat, f = sae(x)
        loss = (xhat - xn).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        fired += (f > 0).any(0).float()
        if (step + 1) % max(1, steps // 5) == 0:
            log(f"    autoencoder step {step + 1}/{steps}: reconstruction error {loss.item():.3f}")
    with torch.no_grad():
        x = hold.to(device)
        xn = (x - sae.mu) / sae.scale
        xhat, _ = sae(x)
        explained = 1 - (xhat - xn).pow(2).sum().item() / (xn - xn.mean(0)).pow(2).sum().item()
    sae.cpu().eval()
    return sae, dict(explained=explained, dead=(fired == 0).float().mean().item(), dropped=dropped)


def fit_weights(T, F, steps=300):
    """Logistic regression on standardized inputs, folded to one direction + threshold.
    T: rows where the digit is present, F: rows where it isn't. Holds out the last 20% of
    each for the reported accuracy."""
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


@torch.no_grad()
def fit_read(sae, states, present, m):
    """One read from the model's own features. states: [lines, d] at line ends; present: bool [lines].
    Picks the m features that best separate the two groups (on the training part only), then
    fits weights and a threshold on those. Returns the read, with its held-out accuracy."""
    F = sae.encode(states.float())
    T, N = F[present], F[~present]
    nT, nN = int(len(T) * 0.8), int(len(N) * 0.8)              # fit_weights holds out the last 20%
    sep = (T[:nT].mean(0) - N[:nN].mean(0)) / (torch.cat([T[:nT], N[:nN]]).std(0) + 1e-6)
    ids = sep.abs().topk(m).indices
    with torch.enable_grad():
        w, thr, acc = fit_weights(T[:, ids], N[:, ids])
    return dict(ids=ids, w=w, thr=thr, acc=acc)
