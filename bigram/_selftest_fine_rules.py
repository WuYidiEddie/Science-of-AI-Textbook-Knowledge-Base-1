"""Rule-only solver v2: fine-grained architecture rules from knowledge_base/07.

Still does NOT read source banks. Implements A-WIDTH / A-DFF / A-HEAD / A-DEPTH
/ A-GRU package logic. Compares to questions.jsonl answers only at scoring time.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(r"D:/research/Science of AI Textbook")
QS = ROOT / "experiments/bigram/questions.jsonl"
PACKS = [ROOT / "v1.5/题包", ROOT / "v1.5/architecture-only"]


def index_materials() -> dict[str, Path]:
    idx = {}
    for root in PACKS:
        if not root.exists():
            continue
        for p in root.rglob("q_*.md"):
            idx.setdefault(p.stem, p)
    return idx


def parse_choice_block(block: str) -> dict:
    def grab(pattern, cast=str):
        m = re.search(pattern, block)
        if not m:
            return None
        s = m.group(1).strip()
        if s in ("None", "none", "null"):
            return None
        try:
            return cast(s)
        except Exception:
            return s

    typ = grab(r"- Type:\s*(.+)")
    d_model = (
        grab(r"- d_model:\s*(\d+)", int)
        or grab(r"- d_model \(embedding and hidden size\):\s*(\d+)", int)
        or grab(r"- Width:\s*(\d+)", int)
    )
    layers = grab(r"- num_layers:\s*(\d+)", int) or grab(r"- Depth:\s*(\d+)", int)
    heads = grab(r"- num_heads:\s*(\d+)", int)
    d_ff = grab(r"- d_ff:\s*(\d+)", int)
    opt = grab(r"- Optimizer:\s*(\w+)")
    lr = grab(r"- Learning rate:\s*([0-9.eE+-]+)", float)
    steps = grab(r"- training_steps:\s*(\d+)", int)
    bs = grab(r"- batch_size:\s*(\d+)", int)
    wd = grab(r"- Weight decay:\s*([0-9.eE+-]+)", float)
    mom = grab(r"- Momentum:\s*([0-9.eE+-]+|None)", float)
    return {
        "type": typ,
        "d_model": d_model,
        "num_layers": layers,
        "num_heads": heads,
        "d_ff": d_ff,
        "opt": opt,
        "lr": lr,
        "steps": steps,
        "bs": bs,
        "wd": wd,
        "mom": mom,
    }


def parse_question(text: str) -> list[dict]:
    parts = re.split(r"^### Choice ", text, flags=re.M)[1:]
    out = []
    for p in parts:
        out.append({"letter": p[0].strip(), **parse_choice_block(p)})
    return out


def is_gru(typ) -> bool:
    return bool(typ and "gru" in typ.lower())


def is_tf(typ) -> bool:
    return bool(typ and ("transformer" in typ.lower() or "causal t" in typ.lower()))


# ---------- OPT / LR (unchanged essence from 01) ----------

def lr_regime(opt, lr):
    if lr is None or opt is None:
        return 0.0, "unknown"
    opt = opt.lower()
    llr = math.log10(lr)
    if opt == "sgd":
        ideal, width = -2.0, 0.7
    elif opt == "adagrad":
        ideal, width = -2.0, 0.8
    elif opt in ("adam", "adamw", "rmsprop"):
        ideal, width = -3.2, 1.0
    else:
        return 0.5, "unknown-opt"
    d = abs(llr - ideal)
    score = max(0.0, 1.0 - (d / width) ** 2)
    if opt in ("sgd", "adagrad"):
        if lr <= 3e-4:
            score = min(score, 0.05)
        elif lr <= 1e-3:
            score = min(score, 0.25)
    else:
        if lr >= 0.03:
            score = min(score, 0.1)
        elif lr >= 0.01:
            score = min(score, 0.35)
        elif lr <= 3e-5:
            score = min(score, 0.15)
        elif lr <= 1e-4:
            score = min(score, 0.55)
    regime = "good" if score >= 0.7 else ("ok" if score >= 0.4 else "bad")
    return score, regime


def opt_base_rank(opt):
    if not opt:
        return 0.5
    return {
        "rmsprop": 1.0,
        "adagrad": 0.85,
        "adamw": 0.75,
        "adam": 0.70,
        "sgd": 0.35,
    }.get(opt.lower(), 0.5)


# ---------- ARCH fine (07) ----------

def width_score(w):
    """A-W1: strongest lever. Bucket by tier, large gaps."""
    if w is None:
        return 0.0
    if w >= 128:
        return 3.0
    if w >= 64:
        return 1.5
    if w >= 32:
        return 0.0
    return -1.5


def dff_score(w, dff):
    """A-D1/A-D2: bigger dff better; diminishing at w>=128."""
    if not dff or dff <= 0:
        return 0.0
    base = {256: 1.6, 128: 1.0, 64: 0.3}.get(dff)
    if base is None:
        # extrapolate mildly
        base = 1.6 if dff >= 256 else (1.0 if dff >= 128 else 0.3)
    if w is not None and w >= 128:
        # A-D2: at w=128, 256 only weakly better than 128
        if dff >= 256:
            base = 1.15
        elif dff >= 128:
            base = 1.0
    return base


def heads_score(w, dff, h):
    """A-H1 conditional table. Small weights; dff=256 near-flat."""
    if h is None:
        return 0.0
    # normalize to 2-vs-4 style
    if w is not None and w <= 32:
        # prefer 4
        return 0.6 if h == 4 else (0.0 if h == 2 else 0.3)
    # w>=64
    if dff == 128:
        # strong 2 heads
        return 0.7 if h == 2 else (0.0 if h == 4 else 0.35)
    if dff == 64:
        return 0.55 if h == 2 else (0.0 if h == 4 else 0.25)
    if dff == 256:
        # A-H1: near flat, slight 2
        return 0.15 if h == 2 else (0.0 if h == 4 else 0.08)
    # default mild 2
    return 0.25 if h == 2 else 0.0


def depth_score(d):
    """A-DP1/A-DP2: weak, prefer 4~3 over 2 over 1, but not strong."""
    if d is None:
        return 0.0
    table = {1: 0.0, 2: 0.15, 3: 0.35, 4: 0.45}
    return table.get(d, 0.2 + 0.05 * min(d, 6))


def kind_score(c):
    """A-G1/A-G2: low lr -> TF; mid-high lr / mid steps -> GRU."""
    lr = c.get("lr")
    steps = c.get("steps")
    if is_gru(c.get("type")):
        s = 0.25
        if lr is not None and 1e-3 <= lr <= 3e-3:
            s += 0.85  # mid lr strongly favors GRU
        elif lr is not None and lr > 3e-3:
            s += 0.5
        if steps is not None and steps <= 512:
            s += 0.25
        if lr is not None and lr <= 3e-4:
            s -= 0.35  # low lr favors TF
        return s
    if is_tf(c.get("type")):
        s = 0.35
        if lr is not None and lr <= 3e-4:
            s += 0.55
        if steps is not None and steps >= 1024 and (lr is None or lr <= 3e-3):
            s += 0.15
        return s
    return 0.3


def package_bonus(c):
    """A-PKG / A-W3: known strong packages get a small nudge; no linear add of heads."""
    w = c.get("d_model")
    d = c.get("num_layers")
    dff = c.get("d_ff")
    h = c.get("num_heads")
    s = 0.0
    # narrow compensation package
    if w == 32 and dff == 256 and h == 4 and d in (3, 4):
        s += 0.4
    # deep-wide-balanced
    if w == 128 and dff == 128 and h == 4 and d == 4:
        s += 0.35
    # shallow-wide-bigFF is high variance: no bonus (or tiny)
    if w == 128 and dff == 256 and h == 2 and d == 1:
        s += 0.05
    # w64 + dff256 is good package
    if w == 64 and dff == 256 and h == 2:
        s += 0.2
    return s


def arch_score_fine(c):
    w = c.get("d_model")
    d = c.get("num_layers")
    dff = c.get("d_ff")
    h = c.get("num_heads")
    # order: width >> dff > heads~depth (from 07 decision order)
    s = 1.0 * width_score(w)
    s += 0.55 * dff_score(w, dff if dff else 0)
    s += 0.35 * heads_score(w, dff if dff else 0, h)
    s += 0.25 * depth_score(d)
    s += package_bonus(c)
    # GRU has no dff/heads: don't zero it out via missing fields — kind_score covers it
    if is_gru(c.get("type")):
        # width still applies; dff/heads should not apply
        s = 1.0 * width_score(w) + 0.25 * depth_score(d) + package_bonus(c)
    return s


def budget_score(c):
    steps = c.get("steps")
    lr = c.get("lr")
    opt = (c.get("opt") or "").lower()
    if steps is None or lr is None:
        return 0.0
    s = 0.0
    if lr <= 3e-4:
        s += min(steps, 1024) / 1024.0 * 1.2
    elif opt == "adam" and 1e-3 <= lr <= 3e-3:
        if steps <= 256:
            s += 1.0
        elif steps <= 512:
            s += 0.5
        elif steps <= 1024:
            s += 0.1
        else:
            s -= 0.3
    else:
        if lr <= 3e-3:
            s += min(steps, 2048) / 2048.0 * 0.6
        else:
            if steps <= 512:
                s += 0.5
            elif steps <= 1024:
                s += 0.2
            else:
                s -= 0.2
    return s


def score_choice(c, qtype: str) -> tuple[float, str]:
    lr_s, regime = lr_regime(c.get("opt"), c.get("lr"))
    opt_s = opt_base_rank(c.get("opt"))
    a_s = arch_score_fine(c)
    k_s = kind_score(c)
    b_s = budget_score(c)

    if qtype == "optimizer_only":
        total = 3.0 * lr_s + 1.5 * opt_s + 0.15 * b_s + 0.1 * k_s
    elif qtype == "architecture_only":
        # architecture shared ctx -> arch dominates; kind matters
        total = 0.35 * lr_s + 0.25 * opt_s + 2.2 * a_s + 0.8 * k_s + 0.15 * b_s
    else:
        total = 2.2 * lr_s + 1.1 * opt_s + 1.4 * a_s + 0.7 * k_s + 0.4 * b_s

    note = (
        f"lr={lr_s:.2f}({regime}) opt={opt_s:.2f} arch={a_s:.2f} "
        f"kind={k_s:.2f} bud={b_s:.2f} "
        f"w={c.get('d_model')} d={c.get('num_layers')} dff={c.get('d_ff')} h={c.get('num_heads')}"
    )
    return total, note


def main():
    materials = index_materials()
    questions = [json.loads(l) for l in QS.open(encoding="utf-8")]
    hit = 0
    by_type = {}
    detailed = []
    for q in questions:
        text = materials[q["qid"]].read_text(encoding="utf-8")
        choices = parse_question(text)
        scored = []
        for c in choices:
            sc, note = score_choice(c, q["type"])
            scored.append((c["letter"], sc, note))
        scored.sort(key=lambda x: -x[1])
        pred = scored[0][0]
        ok = pred == q["answer"]
        hit += int(ok)
        by_type.setdefault(q["type"], {"n": 0, "hit": 0})
        by_type[q["type"]]["n"] += 1
        by_type[q["type"]]["hit"] += int(ok)
        detailed.append(
            {
                "qid": q["qid"],
                "type": q["type"],
                "ans": q["answer"],
                "pred": pred,
                "ok": ok,
                "scores": {s[0]: round(s[1], 3) for s in scored},
                "notes": {s[0]: s[2] for s in scored},
            }
        )

    acc = hit / len(questions)
    print(f"FINE-RULES OVERALL {hit}/{len(questions)} = {acc:.3f}")
    for t, d in sorted(by_type.items()):
        print(f"  {t}: {d['hit']}/{d['n']} = {d['hit']/d['n']:.3f}")

    out = ROOT / "experiments/bigram/_selftest_fine_results.json"
    out.write_text(
        json.dumps(
            {"overall": {"hit": hit, "n": len(questions), "acc": acc}, "by_type": by_type, "items": detailed},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", out)
    misses = [d for d in detailed if not d["ok"]]
    print(f"MISSES {len(misses)}")
    for d in misses[:20]:
        print(d["qid"], d["type"], "ans", d["ans"], "pred", d["pred"], d["scores"])


if __name__ == "__main__":
    main()
