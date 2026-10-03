"""Rule-only solver for bigram questions.

Uses ONLY knowledge_base/bigram conclusions. Does not read experiment_result
or experiments/*/source banks. Scores accuracy against questions.jsonl answers
after predicting.
"""
from __future__ import annotations

import json
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
    vocab = grab(r"- Vocab size:\s*(\d+)", int)
    ctxlen = grab(r"- Context length:\s*(\d+)", int)
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
        "vocab": vocab,
        "ctxlen": ctxlen,
    }


def parse_question(text: str) -> list[dict]:
    parts = re.split(r"^### Choice ", text, flags=re.M)[1:]
    out = []
    for p in parts:
        letter = p[0].strip()
        # stop at model code / later sections if any
        out.append({"letter": letter, **parse_choice_block(p)})
    return out


# ---------- knowledge-base scoring (lower CE better => higher score better) ----------
# We produce a pseudo-CE (lower=better). Only relative within a question matters.

def is_gru(typ: str | None) -> bool:
    return bool(typ and "gru" in typ.lower())


def is_tf(typ: str | None) -> bool:
    return bool(typ and ("transformer" in typ.lower() or typ.lower().startswith("causal t")))


def lr_regime(opt: str | None, lr: float | None) -> tuple[float, str]:
    """B-OPT-04: how suitable is this lr for this optimizer. Higher=better fit."""
    if lr is None or opt is None:
        return 0.0, "unknown"
    opt = opt.lower()
    # log10 lr
    import math

    llr = math.log10(lr)
    # ideal centers
    if opt == "sgd":
        # want ~1e-2
        ideal = -2.0
        width = 0.7  # generous
    elif opt == "adagrad":
        ideal = -2.0
        width = 0.8
    elif opt in ("adam", "adamw", "rmsprop"):
        # 3e-4 ~ 3e-3 center ~ -3.2
        ideal = -3.2
        width = 1.0
    else:
        return 0.5, "unknown-opt"
    # gaussian-ish score in 0..1
    d = abs(llr - ideal)
    score = max(0.0, 1.0 - (d / width) ** 2)
    # extra hard penalties outside sensible band (B-OPT-01: wrong lr ~0.44 CE)
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


def opt_base_rank(opt: str | None) -> float:
    """B-OPT-02 (tuned lr): RMSprop > Adagrad ~ AdamW ~ Adam > SGD."""
    if not opt:
        return 0.5
    table = {
        "rmsprop": 1.0,
        "adagrad": 0.85,
        "adamw": 0.75,
        "adam": 0.70,
        "sgd": 0.35,
    }
    return table.get(opt.lower(), 0.5)


def arch_score(c: dict) -> tuple[float, str]:
    """B-ARCH-01..06. Return (higher=better, note)."""
    s = 0.0
    notes = []
    w = c.get("d_model")
    d = c.get("num_layers")
    heads = c.get("num_heads")
    dff = c.get("d_ff")

    # B-ARCH-01 width: strong
    if w is not None:
        if w >= 128:
            s += 2.0
        elif w >= 64:
            s += 1.0
        elif w >= 32:
            s += 0.0
        else:
            s -= 1.0
        notes.append(f"w={w}")

    # B-ARCH-02 depth: weak
    if d is not None:
        s += 0.3 * (d - 1)  # 1..4
        notes.append(f"d={d}")

    # B-ARCH-03 dff: medium (skip GRU)
    if dff is not None and dff > 0:
        if dff >= 256:
            s += 1.2
        elif dff >= 128:
            s += 0.8
        elif dff >= 64:
            s += 0.3
        notes.append(f"dff={dff}")

    # B-ARCH-04 heads depends on width
    if heads is not None and w is not None:
        if w >= 64:
            # prefer 2 over 4
            if heads == 2:
                s += 1.0
            elif heads == 4:
                s += 0.0
            elif heads == 1:
                s += 0.4
            else:
                s += 0.2
        else:
            # w=32: 4 slightly better than 2 (weak)
            if heads == 4:
                s += 0.2
            elif heads == 2:
                s += 0.0
            else:
                s += 0.1
        notes.append(f"h={heads}")

    return s, ",".join(notes)


def kind_score(c: dict, lr: float | None, steps: int | None) -> float:
    """B-ARCH-05: no universal winner; GRU can win at mid-high lr / short budget."""
    if is_gru(c.get("type")):
        s = 0.2
        # B-ARCH-05 / B-BUD-02 flavor: mid-high lr + shorter budget favors GRU stability
        if lr is not None and lr >= 1e-3:
            s += 0.4
        if steps is not None and steps <= 256:
            s += 0.2
        return s
    if is_tf(c.get("type")):
        s = 0.3
        if lr is not None and lr <= 3e-4:
            s += 0.3  # TF with low lr is fine
        return s
    return 0.25


def budget_score(c: dict) -> float:
    """B-BUD-01/02/03: steps x lr x opt interaction."""
    steps = c.get("steps")
    lr = c.get("lr")
    opt = (c.get("opt") or "").lower()
    if steps is None or lr is None:
        return 0.0
    s = 0.0
    # B-BUD-01: low lr -> more steps better
    if lr <= 3e-4:
        s += min(steps, 1024) / 1024.0 * 1.2
    # B-BUD-02: Adam + mid-high lr -> fewer steps better
    elif opt == "adam" and 1e-3 <= lr <= 3e-3:
        # prefer 256 over 1024/2048
        if steps <= 256:
            s += 1.0
        elif steps <= 512:
            s += 0.5
        elif steps <= 1024:
            s += 0.1
        else:
            s -= 0.3
    # B-BUD-03: general — very short budget is bad at low lr; very long at high lr is risky
    else:
        # neutral-to-slightly prefer more steps if lr moderate
        if lr <= 3e-3:
            s += min(steps, 2048) / 2048.0 * 0.6
        else:
            # high lr: don't overtrain
            if steps <= 512:
                s += 0.5
            elif steps <= 1024:
                s += 0.2
            else:
                s -= 0.2
    # tiny bonus for more samples when nothing else applies
    return s


def score_choice(c: dict, qtype: str) -> tuple[float, str]:
    """Pseudo negative-CE score: higher = better predicted performance."""
    lr = c.get("lr")
    opt = c.get("opt")
    steps = c.get("steps")

    lr_s, regime = lr_regime(opt, lr)
    # B-OPT-01: lr suitability dominates
    opt_s = opt_base_rank(opt)
    a_s, a_note = arch_score(c)
    k_s = kind_score(c, lr, steps)
    b_s = budget_score(c)

    # weights by question type
    if qtype == "optimizer_only":
        # architecture shared -> mostly lr+opt
        total = 3.0 * lr_s + 1.5 * opt_s + 0.15 * b_s + 0.1 * k_s
    elif qtype == "architecture_only":
        # training ctx shared -> mostly arch
        total = 0.4 * lr_s + 0.3 * opt_s + 2.0 * a_s + 0.4 * k_s + 0.2 * b_s
    else:  # mixed
        total = 2.2 * lr_s + 1.2 * opt_s + 1.2 * a_s + 0.5 * k_s + 0.4 * b_s

    note = f"lr={lr_s:.2f}({regime}) opt={opt_s:.2f} arch={a_s:.2f}[{a_note}] kind={k_s:.2f} bud={b_s:.2f}"
    return total, note


def main():
    materials = index_materials()
    questions = [json.loads(l) for l in QS.open(encoding="utf-8")]
    rows = []
    hit = 0
    by_type = {}
    detailed = []

    for q in questions:
        qid = q["qid"]
        qtype = q["type"]
        ans = q["answer"]
        text = materials[qid].read_text(encoding="utf-8")
        choices = parse_question(text)
        if len(choices) < 3:
            # fallback to questions.jsonl raw
            choices = [
                {"letter": c["letter"], **{k: c["raw"].get(k) for k in c["raw"]}}
                for c in q["choices"]
            ]
        scored = []
        for c in choices:
            sc, note = score_choice(c, qtype)
            scored.append((c["letter"], sc, note, c))
        scored.sort(key=lambda x: -x[1])
        pred = scored[0][0]
        ok = pred == ans
        hit += int(ok)
        by_type.setdefault(qtype, {"n": 0, "hit": 0})
        by_type[qtype]["n"] += 1
        by_type[qtype]["hit"] += int(ok)
        detailed.append(
            {
                "qid": qid,
                "type": qtype,
                "ans": ans,
                "pred": pred,
                "ok": ok,
                "scores": {s[0]: round(s[1], 3) for s in scored},
                "notes": {s[0]: s[2] for s in scored},
            }
        )

    acc = hit / len(questions)
    print(f"OVERALL {hit}/{len(questions)} = {acc:.3f}")
    for t, d in sorted(by_type.items()):
        print(f"  {t}: {d['hit']}/{d['n']} = {d['hit']/d['n']:.3f}")

    out = ROOT / "experiments/bigram/_selftest_results.json"
    out.write_text(
        json.dumps(
            {"overall": {"hit": hit, "n": len(questions), "acc": acc}, "by_type": by_type, "items": detailed},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", out)

    # show misses
    misses = [d for d in detailed if not d["ok"]]
    print(f"\nMISSES {len(misses)}")
    for d in misses[:25]:
        print(f"{d['qid']} {d['type']} ans={d['ans']} pred={d['pred']} scores={d['scores']}")


if __name__ == "__main__":
    main()
