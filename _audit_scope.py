"""Audit knowledge_base markdown for explicit applicability scope."""
from __future__ import annotations

import re
from pathlib import Path

KB = Path(r"D:/research/Science of AI Textbook/knowledge_base")

SCOPE_PAT = re.compile(r"适用范围|范围外|边界|Scope|不覆盖|不要外推|协议", re.I)
CONCL_PAT = re.compile(r"^#{2,3}\s+(.*(?:结论|规则|C0|K0|B0|W-|F-|H-|D-|G |R[0-9]|A-).*)$", re.M)
H2_PAT = re.compile(r"^##\s+(.+)$", re.M)


def audit_file(p: Path) -> dict:
    text = p.read_text(encoding="utf-8", errors="replace")
    h2s = H2_PAT.findall(text)
    has_scope_section = bool(re.search(r"适用范围", text))
    has_boundary = bool(re.search(r"边界|范围外|不覆盖|不要外推", text))
    # count rule-like headings
    rules = []
    for m in re.finditer(r"^###?\s+(.+)$", text, re.M):
        title = m.group(1).strip()
        # skip generic section chrome
        if any(x in title for x in ["目录", "参考", "复现", "证据", "写法", "数据来源"]):
            continue
        rules.append(title)
    # For each "### R/W/F/..." block, does nearby text mention scope?
    # Simpler: list ### headings and whether file has any scope near them
    scope_hits = len(SCOPE_PAT.findall(text))
    return {
        "file": str(p.relative_to(KB)),
        "h2": h2s,
        "n_rules": len(rules),
        "rules_sample": rules[:12],
        "has_scope_section": has_scope_section,
        "has_boundary": has_boundary,
        "scope_keyword_hits": scope_hits,
        "len": len(text),
    }


def main():
    rows = []
    for p in sorted(KB.rglob("*.md")):
        if p.name in ("README.md",) and p.parent == KB:
            rows.append(audit_file(p))
            continue
        rows.append(audit_file(p))

    # classify
    missing_scope = []
    missing_boundary = []
    weak = []
    ok = []
    for r in rows:
        if r["file"] == "README.md":
            continue
        if not r["has_scope_section"] and not r["has_boundary"]:
            missing_scope.append(r)
        elif not r["has_scope_section"]:
            missing_scope.append(r)
        elif not r["has_boundary"]:
            missing_boundary.append(r)
        elif r["scope_keyword_hits"] < 3:
            weak.append(r)
        else:
            ok.append(r)

    print(f"total md: {len(rows)}")
    print(f"OK (scope+boundary): {len(ok)}")
    print(f"\n=== MISSING 适用范围 section: {len(missing_scope)} ===")
    for r in missing_scope:
        print(f"  {r['file']}  rules~{r['n_rules']}  hits={r['scope_keyword_hits']}  h2={r['h2'][:6]}")
    print(f"\n=== HAS scope but NO 边界: {len(missing_boundary)} ===")
    for r in missing_boundary:
        print(f"  {r['file']}  hits={r['scope_keyword_hits']}  h2={r['h2'][:6]}")
    print(f"\n=== WEAK (few scope keywords): {len(weak)} ===")
    for r in weak:
        print(f"  {r['file']}  hits={r['scope_keyword_hits']}  h2={r['h2'][:8]}")


if __name__ == "__main__":
    main()
