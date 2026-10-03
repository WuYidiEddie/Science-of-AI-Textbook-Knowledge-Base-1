"""Table-driven solver: uses ONLY numeric tables from knowledge_base 08/09/10.

Implements B08-1 gap table, B08-3 steps, B09-1 arch deltas, B10-1 protocol.
Does not read source banks.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(r"D:/research/Science of AI Textbook")
QS = ROOT / "experiments/bigram/questions.jsonl"
PACKS = [ROOT / "v1.5/题包", ROOT / "v1.5/architecture-only"]

# B08-1: median gap vs cell-best (lower better)
GAP = {
    ("Adagrad", 0.01): 0.012,
    ("RMSprop", 0.001): 0.069,
    ("Adam", 0.01): 0.070,
    ("AdamW", 0.01): 0.076,
    ("Adam", 0.003): 0.089,
    ("AdamW", 0.003): 0.089,
    ("Adam", 0.001): 0.091,
    ("AdamW", 0.001): 0.088,
    ("RMSprop", 0.003): 0.081,
    ("RMSprop", 0.0003): 0.091,
    ("Adagrad", 0.003): 0.095,
    ("Adam", 0.0003): 0.167,
    ("AdamW", 0.0003): 0.168,
    ("RMSprop", 0.01): 0.270,
    ("Adagrad", 0.001): 0.268,
    ("RMSprop", 0.0001): 0.249,
    ("Adam", 0.0001): 0.291,
    ("AdamW", 0.0001): 0.291,
    ("Adagrad", 0.0003): 0.365,
    ("SGD", 0.01): 0.235,
    ("SGD", 0.03): 0.261,
    ("SGD", 0.003): 0.321,
    ("SGD", 0.001): 0.379,
    ("Adagrad", 0.0001): 0.417,
    ("Adagrad", 3e-05): 0.442,
    ("SGD", 0.0003): 0.407,
    ("SGD", 0.0001): 0.426,
    ("SGD", 3e-05): 0.465,
    ("Adam", 3e-05): 0.359,
    ("AdamW", 3e-05): 0.359,
    ("RMSprop", 3e-05): 0.334,
}


def index_materials():
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

    return {
        "type": grab(r"- Type:\s*(.+)"),
        "d_model": grab(r"- d_model:\s*(\d+)", int)
        or grab(r"- d_model \(embedding and hidden size\):\s*(\d+)", int)
        or grab(r"- Width:\s*(\d+)", int),
        "num_layers": grab(r"- num_layers:\s*(\d+)", int) or grab(r"- Depth:\s*(\d+)", int),
        "num_heads": grab(r"- num_heads:\s*(\d+)", int),
        "d_ff": grab(r"- d_ff:\s*(\d+)", int),
        "opt": grab(r"- Optimizer:\s*(\w+)"),
        "lr": grab(r"- Learning rate:\s*([0-9.eE+-]+)", float),
        "steps": grab(r"- training_steps:\s*(\d+)", int),
        "bs": grab(r"- batch_size:\s*(\d+)", int),
    }


def parse_question(text: str):
    parts = re.split(r"^### Choice ", text, flags=re.M)[1:]
    return [{"letter": p[0].strip(), **parse_choice_block(p)} for p in parts]


def is_gru(typ) -> bool:
    return bool(typ and "gru" in typ.lower())


def gap_opt_lr(opt, lr) -> float:
    """B08-1 lookup with mild interpolation for unseen points."""
    if opt is None or lr is None:
        return 0.25
    opt = opt[0].upper() + opt[1:].lower()
    if opt == "Adamw":
        opt = "AdamW"
    # exact
    for (o, l), g in GAP.items():
        if o.lower() == opt.lower() and abs(l - lr) / max(l, lr) < 0.02:
            return g
    # nearest in log-lr same opt
    cands = [(l, g) for (o, l), g in GAP.items() if o.lower() == opt.lower()]
    if not cands:
        return 0.25
    llr = math.log10(lr)
    best = min(cands, key=lambda t: abs(math.log10(t[0]) - llr))
    return best[1]


def steps_delta(steps, lr) -> float:
    """B08-3: contribution to CE relative to steps=256 baseline. negative better."""
    if steps is None or lr is None:
        return 0.0
    # from table median(CE(s)-CE(256))
    # approximate by lr band
    if lr <= 3e-4:
        # more steps better
        table = {64: 0.14, 256: 0.0, 512: -0.05, 1024: -0.14, 2048: -0.16}
    elif lr >= 2e-3:
        # mid-high: fewer steps better at 3e-3 especially
        table = {64: -0.05, 256: 0.0, 512: -0.1, 1024: 0.37, 2048: 0.45}
    else:
        table = {64: 0.08, 256: 0.0, 512: -0.02, 1024: 0.04, 2048: 0.08}
    if steps in table:
        return table[steps]
    # interpolate
    keys = sorted(table)
    if steps < keys[0]:
        return table[keys[0]]
    if steps > keys[-1]:
        return table[keys[-1]]
    return 0.0


def arch_delta(c) -> float:
    """B09-1 sum of median deltas vs a reference. Lower CE better => we return penalty.

    Reference: d=2, w=64, dff=64, h=2, tf.
    """
    w = c.get("d_model")
    d = c.get("num_layers")
    dff = c.get("d_ff")
    h = c.get("num_heads")
    if is_gru(c.get("type")):
        # gru package roughly +0.05 vs strong TF wide, -0.05 vs weak
        return 0.05

    pen = 0.0
    # width tier relative to w=64
    if w is not None:
        if w >= 128:
            pen += -0.077  # 128 better than 64
        elif w <= 32:
            pen += +0.061  # 32 worse than 64
    # dff
    if dff:
        if w is not None and w <= 64:
            if dff >= 256:
                pen += -0.044
            elif dff >= 128:
                pen += -0.020
        elif w is not None and w >= 128:
            if dff >= 256:
                pen += -0.011
            elif dff >= 128:
                pen += -0.011
            if dff <= 64:
                pen += +0.025
        else:
            if dff >= 256:
                pen += -0.03
            elif dff >= 128:
                pen += -0.015
    # heads: ONLY when dff=128 and w>=64 (B09-1c: dff=256 -> 0)
    if h is not None and w is not None:
        if dff == 128 and w >= 64:
            if h == 2:
                pen += -0.015
            elif h == 4:
                pen += 0.0
        elif dff == 256:
            pen += 0.0  # knife-edge
        elif w <= 32:
            if h == 4:
                pen += -0.01
        else:
            if h == 2:
                pen += -0.005
    # depth weak
    if d is not None:
        if d >= 4:
            pen += -0.02
        elif d == 1:
            pen += +0.01
    return pen


def estimate_ce(c) -> float:
    """Lower predicted CE is better. Components from B08/B09."""
    g = gap_opt_lr(c.get("opt"), c.get("lr"))
    s = steps_delta(c.get("steps"), c.get("lr"))
    a = arch_delta(c)
    # tiny GRU context bonus at mid lr (A-G1) as epsilon
    kind_adj = 0.0
    if is_gru(c.get("type")):
        lr = c.get("lr")
        if lr is not None and 1e-3 <= lr <= 3e-3:
            kind_adj = -0.08
        elif lr is not None and lr <= 3e-4:
            kind_adj = +0.05
    return g + s + a + kind_adj


def main():
    materials = index_materials()
    questions = [json.loads(l) for l in QS.open(encoding="utf-8")]
    hit = 0
    by_type = {}
    detailed = []
    for q in questions:
        choices = parse_question(materials[q["qid"]].read_text(encoding="utf-8"))
        ests = []
        for c in choices:
            e = estimate_ce(c)
            ests.append((c["letter"], e, c))
        ests.sort(key=lambda x: x[1])
        pred = ests[0][0]
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
                "ests": {e[0]: round(e[1], 4) for e in ests},
            }
        )

    acc = hit / len(questions)
    print(f"TABLE-DRIVEN OVERALL {hit}/{len(questions)} = {acc:.3f}")
    for t, d in sorted(by_type.items()):
        print(f"  {t}: {d['hit']}/{d['n']} = {d['hit']/d['n']:.3f}")

    # subset analysis B10-1 predictions
    easy = hard = easy_h = hard_h = 0
    for q, det in zip(questions, detailed):
        gaps = [gap_opt_lr(
            next(c for c in parse_question(materials[q['qid']].read_text(encoding='utf-8')) if c['letter']==L).get('opt'),
            next(c for c in parse_question(materials[q['qid']].read_text(encoding='utf-8')) if c['letter']==L).get('lr'),
        ) for L in 'ABC']
        # cheaper: use ests spread
        spread = max(det['ests'].values()) - min(det['ests'].values())
        if spread >= 0.20:
            easy += 1
            easy_h += int(det['ok'])
        else:
            hard += 1
            hard_h += int(det['ok'])
    print(f"  spread>=0.20: {easy_h}/{easy} = {easy_h/easy if easy else None:.3f}")
    print(f"  spread<0.20:  {hard_h}/{hard} = {hard_h/hard if hard else None:.3f}")

    out = ROOT / "experiments/bigram/_selftest_table_results.json"
    out.write_text(
        json.dumps({"overall": {"hit": hit, "n": len(questions), "acc": acc}, "by_type": by_type, "items": detailed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("wrote", out)
    for d in detailed:
        if not d["ok"] and d["type"] == "architecture_only":
            print("MISS", d["qid"], "ans", d["ans"], "pred", d["pred"], d["ests"])


if __name__ == "__main__":
    main()
