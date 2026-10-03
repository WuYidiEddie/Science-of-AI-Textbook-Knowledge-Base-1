import json, re
from pathlib import Path
from collections import Counter

ROOT = Path(r'D:/research/Science of AI Textbook')
res = json.loads((ROOT/'experiments/bigram/_selftest_results.json').read_text(encoding='utf-8'))

idx = {}
for root in [ROOT/'v1.5/题包', ROOT/'v1.5/architecture-only']:
    for p in root.rglob('q_*.md'):
        idx.setdefault(p.stem, p)

def parse_choice_block(block):
    def grab(pattern, cast=str):
        m = re.search(pattern, block)
        if not m:
            return None
        s = m.group(1).strip()
        if s in ('None', 'none', 'null'):
            return None
        try:
            return cast(s)
        except Exception:
            return s
    return {
        'type': grab(r'- Type:\s*(.+)'),
        'w': grab(r'- d_model:\s*(\d+)', int) or grab(r'- Width:\s*(\d+)', int),
        'd': grab(r'- num_layers:\s*(\d+)', int) or grab(r'- Depth:\s*(\d+)', int),
        'h': grab(r'- num_heads:\s*(\d+)', int),
        'dff': grab(r'- d_ff:\s*(\d+)', int),
        'opt': grab(r'- Optimizer:\s*(\w+)'),
        'lr': grab(r'- Learning rate:\s*([0-9.eE+-]+)', float),
        'steps': grab(r'- training_steps:\s*(\d+)', int),
    }

def parse_q(text):
    parts = re.split(r'^### Choice ', text, flags=re.M)[1:]
    out = {}
    for p in parts:
        out[p[0].strip()] = parse_choice_block(p)
    return out

misses = [d for d in res['items'] if not d['ok'] and d['type'] == 'architecture_only']
feature_bias = Counter()
details = []
for d in misses:
    qid = d['qid']
    ch = parse_q(idx[qid].read_text(encoding='utf-8'))
    a, p = ch[d['ans']], ch[d['pred']]
    for key in ['w', 'd', 'h', 'dff']:
        if a.get(key) is None or p.get(key) is None:
            continue
        if a[key] > p[key]:
            feature_bias['ans_' + key + '_higher'] += 1
        elif a[key] < p[key]:
            feature_bias['pred_' + key + '_higher'] += 1
        else:
            feature_bias['tie_' + key] += 1
    ak = 'gru' if a.get('type') and 'gru' in a['type'].lower() else 'tf'
    pk = 'gru' if p.get('type') and 'gru' in p['type'].lower() else 'tf'
    if ak != pk:
        feature_bias['ans_%s_vs_pred_%s' % (ak, pk)] += 1
    details.append((qid, d['ans'], d['pred'], a, p))

print('arch misses', len(misses))
print('feature bias among misses:')
for k, v in feature_bias.most_common():
    print(' ', k, v)

print('\nsample miss configs (ans | pred):')
for qid, ans, pred, a, p in details[:15]:
    def fmt(x):
        return 'type=%s d=%s w=%s h=%s dff=%s opt=%s lr=%s' % (
            (x.get('type') or '?')[:16], x.get('d'), x.get('w'), x.get('h'), x.get('dff'), x.get('opt'), x.get('lr'))
    print(qid, ans, 'vs', pred)
    print('  ANS', fmt(a))
    print('  PRD', fmt(p))

print('\n=== optimizer misses ===')
om = [d for d in res['items'] if not d['ok'] and d['type'] == 'optimizer_only']
lr_bias = Counter()
opt_bias = Counter()
for d in om:
    ch = parse_q(idx[d['qid']].read_text(encoding='utf-8'))
    a, p = ch[d['ans']], ch[d['pred']]
    if a.get('lr') and p.get('lr'):
        if a['lr'] > p['lr']:
            lr_bias['ans_higher_lr'] += 1
        elif a['lr'] < p['lr']:
            lr_bias['pred_higher_lr'] += 1
        else:
            lr_bias['tie_lr'] += 1
    if a.get('opt') != p.get('opt'):
        opt_bias['ans_%s_pred_%s' % (a.get('opt'), p.get('opt'))] += 1
print('lr bias', dict(lr_bias))
print('opt bias', opt_bias.most_common())
for d in om[:12]:
    ch = parse_q(idx[d['qid']].read_text(encoding='utf-8'))
    a, p = ch[d['ans']], ch[d['pred']]
    print(d['qid'], 'ans', a.get('opt'), a.get('lr'), '| pred', p.get('opt'), p.get('lr'), '| scores', d['scores'])
