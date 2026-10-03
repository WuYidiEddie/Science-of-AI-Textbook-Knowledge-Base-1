"""Fine-grained architecture × training-context grid on XOR.

Goal: falsifiable rules like "under (opt,lr,steps,bs) prefer arch A over B".

Writes arch_context_grid.jsonl next to this script (append, resume-safe).
"""
from __future__ import annotations

import json
import math
import statistics as stats
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(
    r"D:\research\Science of AI Textbook\v1.5\v15-500q-6d4ae9b\data\datasets\xor_classification"
)
OUT = Path(__file__).resolve().parent
RESULTS = OUT / "arch_context_grid.jsonl"
SEEDS = [0, 1, 2]

# two representative instances: dim2 and dim4
INSTS = ["xorcls_1c4cf2", "xorcls_0c0858"]


def _act(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
    }.get(name, nn.ReLU())


class MLPBlock(nn.Module):
    def __init__(self, width, use_ln, residual, act):
        super().__init__()
        self.norm = nn.LayerNorm(width) if use_ln else nn.Identity()
        self.linear = nn.Linear(width, width)
        self.act = act
        self.residual = residual

    def forward(self, x):
        h = self.linear(self.norm(x))
        if self.residual:
            h = h + x
        return self.act(h)


def build_model(in_dim, depth, width, mask, residual, act):
    blocks = []
    for i in range(depth):
        use_ln = bool(mask[i]) if i < len(mask) else False
        blocks.append(MLPBlock(width, use_ln, residual, _act(act)))
    return nn.Sequential(nn.Linear(in_dim, width), _act(act), *blocks, nn.Linear(width, 2))


def make_optimizer(name, params, lr, wd, mom):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(mom or 0.0), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=float(mom or 0.0), weight_decay=wd)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


def load_split(inst):
    p = ROOT / inst
    tr = torch.load(p / "train.pt", map_location="cpu", weights_only=False)
    te = torch.load(p / "test.pt", map_location="cpu", weights_only=False)
    return (
        tr["x"].float().to(DEVICE),
        tr["y"].to(DEVICE),
        te["x"].float().to(DEVICE),
        te["y"].to(DEVICE),
    )


def train_one(tx, ty, vx, vy, arch, opt, lr, wd, mom, steps, bs, seed):
    torch.manual_seed(seed)
    model = build_model(
        tx.shape[1], arch["depth"], arch["width"], arch["mask"], arch["res"], arch["act"]
    ).to(DEVICE)
    optim = make_optimizer(opt, model.parameters(), lr, wd, mom)
    n = tx.shape[0]
    y = ty.reshape(-1).long()
    sy = vy.reshape(-1).long()
    failed = False
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, n, (bs,), device=DEVICE)
        loss = F.cross_entropy(model(tx[idx]), y[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    model.eval()
    with torch.inference_mode():
        metric = float(F.cross_entropy(model(vx), sy).item())
    if not math.isfinite(metric):
        failed = True
    return failed, metric


# architecture tags: keep factorially interpretable
ARCHS = [
    # capacity corners
    dict(tag="d1w32", depth=1, width=32, mask=[0], res=False, act="relu"),
    dict(tag="d1w128", depth=1, width=128, mask=[0], res=False, act="relu"),
    dict(tag="d2w32", depth=2, width=32, mask=[0, 0], res=False, act="relu"),
    dict(tag="d2w64", depth=2, width=64, mask=[0, 0], res=False, act="relu"),
    dict(tag="d2w128", depth=2, width=128, mask=[0, 0], res=False, act="relu"),
    dict(tag="d3w64", depth=3, width=64, mask=[0, 0, 0], res=False, act="relu"),
    dict(tag="d3w128", depth=3, width=128, mask=[0, 0, 0], res=False, act="relu"),
    dict(tag="d4w32", depth=4, width=32, mask=[0, 0, 0, 0], res=False, act="relu"),
    dict(tag="d4w64", depth=4, width=64, mask=[0, 0, 0, 0], res=False, act="relu"),
    dict(tag="d4w128", depth=4, width=128, mask=[0, 0, 0, 0], res=False, act="relu"),
    # LN / residual / act on the deep-wide base
    dict(tag="d4w128_ln", depth=4, width=128, mask=[1, 1, 1, 1], res=False, act="relu"),
    dict(tag="d4w128_res", depth=4, width=128, mask=[0, 0, 0, 0], res=True, act="relu"),
    dict(tag="d4w128_lnres", depth=4, width=128, mask=[1, 1, 1, 1], res=True, act="relu"),
    dict(tag="d4w128_lnres_gelu", depth=4, width=128, mask=[1, 1, 1, 1], res=True, act="gelu"),
    dict(tag="d2w64_lnres", depth=2, width=64, mask=[1, 1], res=True, act="relu"),
    dict(tag="d2w64_lnres_gelu", depth=2, width=64, mask=[1, 1], res=True, act="gelu"),
    # shallow but LN+res (typical question winner)
    dict(tag="d1w64_lnres", depth=1, width=64, mask=[1], res=True, act="relu"),
]

# training contexts
# P1: arch × (opt, lr) at steps=1024, bs=32
CTX_LR = [
    dict(group="P1", opt="SGD", lr=3e-4, steps=1024, bs=32, mom=0.0, wd=0.0),
    dict(group="P1", opt="SGD", lr=3e-3, steps=1024, bs=32, mom=0.0, wd=0.0),
    dict(group="P1", opt="SGD", lr=3e-2, steps=1024, bs=32, mom=0.0, wd=0.0),
    dict(group="P1", opt="Adam", lr=3e-4, steps=1024, bs=32, mom=None, wd=0.0),
    dict(group="P1", opt="Adam", lr=1e-3, steps=1024, bs=32, mom=None, wd=0.0),
    dict(group="P1", opt="Adam", lr=3e-3, steps=1024, bs=32, mom=None, wd=0.0),
    dict(group="P1", opt="RMSprop", lr=3e-3, steps=1024, bs=32, mom=None, wd=0.0),
    dict(group="P1", opt="Adagrad", lr=3e-2, steps=1024, bs=32, mom=None, wd=0.0),
]

# P2: arch × steps on two representative contexts
CTX_STEPS = []
for st in [256, 512, 1024, 2048, 4096]:
    CTX_STEPS.append(dict(group="P2", opt="Adam", lr=3e-3, steps=st, bs=32, mom=None, wd=0.0))
    CTX_STEPS.append(dict(group="P2", opt="SGD", lr=3e-2, steps=st, bs=32, mom=0.0, wd=0.0))

# P3: arch × bs
CTX_BS = []
for bs in [16, 32, 64, 128]:
    CTX_BS.append(dict(group="P3", opt="Adam", lr=3e-3, steps=1024, bs=bs, mom=None, wd=0.0))
    CTX_BS.append(dict(group="P3", opt="SGD", lr=3e-2, steps=1024, bs=bs, mom=0.0, wd=0.0))


def key_of(inst, arch, ctx):
    return (
        inst,
        arch["tag"],
        ctx["group"],
        ctx["opt"],
        ctx["lr"],
        ctx["steps"],
        ctx["bs"],
        ctx.get("mom"),
    )


def load_done():
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["inst"], r["arch_tag"], r["group"], r["opt"], r["lr"], r["steps"], r["bs"], r.get("mom")))
    return done


def main(which="all"):
    print("device", DEVICE, "which", which, flush=True)
    cache = {}
    for inst in INSTS:
        cache[inst] = load_split(inst)
    done = load_done()
    print("already done", len(done), flush=True)
    fo = RESULTS.open("a", encoding="utf-8")

    def run(arch, ctx, inst):
        k = key_of(inst, arch, ctx)
        if k in done:
            return
        t0 = time.time()
        fins, failed = [], 0
        tx, ty, vx, vy = cache[inst]
        for seed in SEEDS:
            f, m = train_one(
                tx, ty, vx, vy, arch, ctx["opt"], ctx["lr"], ctx.get("wd", 0.0),
                ctx.get("mom"), ctx["steps"], ctx["bs"], seed,
            )
            if f:
                failed += 1
            else:
                fins.append(m)
        mean = stats.mean(fins) if fins else float("nan")
        row = {
            "inst": inst,
            "arch_tag": arch["tag"],
            "depth": arch["depth"],
            "width": arch["width"],
            "mask": arch["mask"],
            "res": arch["res"],
            "act": arch["act"],
            "group": ctx["group"],
            "opt": ctx["opt"],
            "lr": ctx["lr"],
            "steps": ctx["steps"],
            "bs": ctx["bs"],
            "mom": ctx.get("mom"),
            "wd": ctx.get("wd", 0.0),
            "finals": fins,
            "final": mean,
            "failed": failed,
            "nseed": len(fins),
        }
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        fo.flush()
        done.add(k)
        print(
            f"{ctx['group']} {inst} {arch['tag']:18s} {ctx['opt']:7s} lr={ctx['lr']:g} "
            f"st={ctx['steps']:5d} bs={ctx['bs']:3d} final={mean:.4f} t={time.time()-t0:.1f}s",
            flush=True,
        )

    if which in ("all", "P1"):
        print("\n===== P1 arch × (opt,lr) =====", flush=True)
        for ctx in CTX_LR:
            for arch in ARCHS:
                for inst in INSTS:
                    run(arch, ctx, inst)

    if which in ("all", "P2"):
        print("\n===== P2 arch × steps =====", flush=True)
        # only a few archs to keep size sane
        subset = [a for a in ARCHS if a["tag"] in {
            "d1w32", "d2w64", "d4w128", "d4w128_lnres", "d2w64_lnres_gelu", "d1w64_lnres"
        }]
        for ctx in CTX_STEPS:
            for arch in subset:
                for inst in INSTS:
                    run(arch, ctx, inst)

    if which in ("all", "P3"):
        print("\n===== P3 arch × bs =====", flush=True)
        subset = [a for a in ARCHS if a["tag"] in {
            "d2w64", "d4w128", "d4w128_lnres", "d2w64_lnres_gelu"
        }]
        for ctx in CTX_BS:
            for arch in subset:
                for inst in INSTS:
                    run(arch, ctx, inst)

    fo.close()
    print("DONE", which, flush=True)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "all")
