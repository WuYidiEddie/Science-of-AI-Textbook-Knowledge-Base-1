"""Faster supplemental runs: incremental JSONL, 3 seeds, resume support."""
from __future__ import annotations

import json
import math
import time
import statistics as stats
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(
    r"D:\research\Science of AI Textbook\v1.5\v15-500q-6d4ae9b\data\datasets\xor_classification"
)
OUT = Path(r"D:\research\Science of AI Textbook\experiments\xor\_analysis_tmp")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "supp_results.jsonl"


def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    return {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }.get(name, nn.ReLU())


class MLPBlock(nn.Module):
    def __init__(self, width: int, use_ln: bool, residual: bool, act: nn.Module):
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


def build_model(in_dim, out_dim, depth, width, mask, residual, act):
    blocks = []
    for i in range(depth):
        use_ln = bool(mask[i]) if i < len(mask) else False
        blocks.append(MLPBlock(width, use_ln, residual, _act(act)))
    net = [nn.Linear(in_dim, width), _act(act)] + blocks + [nn.Linear(width, out_dim)]
    return nn.Sequential(*net)


def make_optimizer(name, params, lr, wd, mom):
    name = (name or "Adam").strip().lower()
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


def load_split(inst_name: str):
    p = ROOT / inst_name
    train = torch.load(p / "train.pt", map_location="cpu", weights_only=False)
    test = torch.load(p / "test.pt", map_location="cpu", weights_only=False)
    return (
        train["x"].float().to(DEVICE),
        train["y"].to(DEVICE),
        test["x"].float().to(DEVICE),
        test["y"].to(DEVICE),
    )


def train_one(train_x, train_y, test_x, test_y, arch, opt, lr, wd, mom, steps, batch_size, seed):
    torch.manual_seed(seed)
    model = build_model(
        in_dim=train_x.shape[1],
        out_dim=2,
        depth=arch["depth"],
        width=arch["width"],
        mask=arch["mask"],
        residual=arch["res"],
        act=arch["act"],
    ).to(DEVICE)
    optimizer = make_optimizer(opt, model.parameters(), lr, wd, mom)
    n = train_x.shape[0]
    ty = train_y.reshape(-1).long()
    sy = test_y.reshape(-1).long()
    failed = False
    for _ in range(steps):
        model.train()
        idx = torch.randint(0, n, (batch_size,), device=DEVICE)
        pred = model(train_x[idx])
        loss = F.cross_entropy(pred, ty[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(test_x)
        metric = float(F.cross_entropy(logits, sy).item())
    if not math.isfinite(metric):
        failed = True
    return failed, metric


def pick_instances():
    insts = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and (p / "train.pt").exists())
    by_dim = defaultdict(list)
    for name in insts:
        x = torch.load(ROOT / name / "train.pt", map_location="cpu", weights_only=False)["x"]
        by_dim[int(x.shape[1])].append(name)
    picked = []
    for d in [2, 4, 8, 16]:
        if by_dim.get(d):
            picked.append(by_dim[d][0])
    return picked


ARCHS = [
    dict(depth=2, width=32, mask=[0, 0], res=False, act="relu", tag="shallow_narrow"),
    dict(depth=4, width=64, mask=[0, 0, 1, 0], res=True, act="gelu", tag="deep_ln_res"),
]
SEEDS = [0, 1, 2]


def load_done():
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add(
                (
                    r["exp"],
                    r["inst"],
                    r["arch_tag"],
                    r["opt"],
                    r["lr"],
                    r["steps"],
                    r["bs"],
                    tuple(r["mask"]) if isinstance(r["mask"], list) else r["mask"],
                    r["res"],
                    r["act"],
                    r["d"],
                    r["w"],
                )
            )
    return done


def main(which: str = "all"):
    print("device", DEVICE, "which", which, flush=True)
    picked = pick_instances()
    print("picked", picked, flush=True)
    done = load_done()
    print("already done", len(done), flush=True)
    cache = {}

    def get(inst):
        if inst not in cache:
            cache[inst] = load_split(inst)
        return cache[inst]

    fo = RESULTS.open("a", encoding="utf-8")

    def run(exp, **kw):
        key = (
            exp,
            kw["inst"],
            kw["arch"].get("tag", ""),
            kw["opt"],
            kw["lr"],
            kw["steps"],
            kw["bs"],
            tuple(kw["arch"]["mask"]),
            kw["arch"]["res"],
            kw["arch"]["act"],
            kw["arch"]["depth"],
            kw["arch"]["width"],
        )
        if key in done:
            print("skip", key, flush=True)
            return
        fins, failed = [], 0
        t0 = time.time()
        for seed in SEEDS:
            f, m = train_one(
                kw["tx"], kw["ty"], kw["vx"], kw["vy"],
                arch=kw["arch"], opt=kw["opt"], lr=kw["lr"],
                wd=kw.get("wd", 0.0), mom=kw.get("mom"),
                steps=kw["steps"], batch_size=kw["bs"], seed=seed,
            )
            if f:
                failed += 1
            else:
                fins.append(m)
        mean = stats.mean(fins) if fins else float("nan")
        row = {
            "exp": exp, "inst": kw["inst"], "arch_tag": kw["arch"].get("tag", ""),
            "d": kw["arch"]["depth"], "w": kw["arch"]["width"],
            "mask": kw["arch"]["mask"], "res": kw["arch"]["res"], "act": kw["arch"]["act"],
            "opt": kw["opt"], "lr": kw["lr"], "wd": kw.get("wd", 0.0), "mom": kw.get("mom"),
            "steps": kw["steps"], "bs": kw["bs"],
            "finals": fins, "final": mean, "failed": failed, "nseed": len(fins),
        }
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        fo.flush()
        done.add(key)
        dt = time.time() - t0
        print(
            f"{exp} {kw['inst']} {row['arch_tag']:16s} opt={kw['opt']:8s} lr={kw['lr']:g} "
            f"st={kw['steps']:5d} final={mean:.4f} t={dt:.1f}s",
            flush=True,
        )

    if which in ("all", "A"):
        print("\n===== A. LR sweep =====", flush=True)
        for inst in picked:
            tx, ty, vx, vy = get(inst)
            for arch in ARCHS:
                for lr in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
                    run("A_lr", inst=inst, arch=arch, opt="Adam", lr=lr, steps=256, bs=32,
                        tx=tx, ty=ty, vx=vx, vy=vy)

    if which in ("all", "B"):
        print("\n===== B. Steps sweep =====", flush=True)
        for inst in picked:
            tx, ty, vx, vy = get(inst)
            arch = ARCHS[1]
            for steps in [256, 512, 1024, 2048, 4096]:
                run("B_steps", inst=inst, arch=arch, opt="Adam", lr=3e-4, steps=steps, bs=32,
                    tx=tx, ty=ty, vx=vx, vy=vy)

    if which in ("all", "C"):
        print("\n===== C. Optimizer sweep =====", flush=True)
        for inst in picked[:2]:
            tx, ty, vx, vy = get(inst)
            arch = ARCHS[1]
            for lr in [3e-4, 3e-3]:
                for opt in ["SGD", "Adam", "AdamW", "RMSprop", "Adagrad"]:
                    mom = 0.0 if opt == "SGD" else None
                    run("C_opt", inst=inst, arch=arch, opt=opt, lr=lr, steps=512, bs=32, mom=mom,
                        tx=tx, ty=ty, vx=vx, vy=vy)

    if which in ("all", "D"):
        print("\n===== D. Arch one-at-a-time =====", flush=True)
        variants = [
            dict(depth=2, width=48, mask=[0, 0], res=False, act="relu", tag="base"),
            dict(depth=1, width=48, mask=[0], res=False, act="relu", tag="d1"),
            dict(depth=3, width=48, mask=[0, 0, 0], res=False, act="relu", tag="d3"),
            dict(depth=2, width=16, mask=[0, 0], res=False, act="relu", tag="w16"),
            dict(depth=2, width=128, mask=[0, 0], res=False, act="relu", tag="w128"),
            dict(depth=2, width=48, mask=[1, 0], res=False, act="relu", tag="ln1"),
            dict(depth=2, width=48, mask=[1, 1], res=False, act="relu", tag="ln2"),
            dict(depth=2, width=48, mask=[0, 0], res=True, act="relu", tag="res"),
            dict(depth=2, width=48, mask=[0, 0], res=False, act="gelu", tag="gelu"),
            dict(depth=2, width=48, mask=[0, 0], res=False, act="silu", tag="silu"),
            dict(depth=2, width=48, mask=[0, 0], res=False, act="leaky_relu", tag="leaky"),
        ]
        for inst in picked[:2]:
            tx, ty, vx, vy = get(inst)
            for arch in variants:
                run("D_arch", inst=inst, arch=arch, opt="Adam", lr=3e-4, steps=512, bs=32,
                    tx=tx, ty=ty, vx=vx, vy=vy)

    fo.close()
    print("DONE", which, flush=True)


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(which)
