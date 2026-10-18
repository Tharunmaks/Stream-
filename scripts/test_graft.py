#!/usr/bin/env python3
"""Graft end to end: teach a frozen model made-up facts, check it learned them and kept its general knowledge.

    make es_graft es_chat && python3 scripts/test_graft.py MODEL.gguf      (a small chat model, e.g. Qwen2.5-0.5B-Instruct)

1. selftest: every gradient against finite differences; the exact loss against full cross-entropy and KL.
2. read + train on 30 made-up companies (4 facts each, 3 question wordings learned, the 4th held out).
3. check: the engine runs the graft exactly as training computed it (batched reading vs one-token-at-a-time generation).
4. the held-out wording of every question, answered greedily: right facts with the graft vs without.
5. general questions (not in the data, not in the built-in anchors): the graft must keep the base model's answers.
"""
import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]): sys.exit(__doc__)
MODEL = os.path.abspath(sys.argv[1])
G, CHAT = os.path.join(ROOT, 'es_graft'), os.path.join(ROOT, 'es_chat')
bad = 0


def check(name, ok, info=''):
    global bad
    bad += not ok
    print('%s %s  %s' % ('PASS' if ok else 'FAIL', name, info), flush=True)


def facts(dirn):
    random.seed(7)
    syl = ['vel', 'mor', 'bri', 'tan', 'qua', 'lix', 'dor', 'ven', 'sha', 'rel', 'pom', 'kai', 'zur', 'fen', 'lo', 'nar', 'tis', 'ob', 'ry', 'gal']
    name = lambda n: ''.join(random.choice(syl) for _ in range(n)).capitalize()
    ents, used = [], set()
    while len(ents) < 30:
        e = name(3)
        if e in used: continue
        used.add(e)
        ents.append(dict(e=e, city=name(2) + random.choice(['haven', 'ford', 'mere', 'burg', 'ton']), founder=name(2) + ' ' + name(3),
                         year=str(random.randint(1952, 2021)),
                         product=random.choice(['solar water filters', 'wooden drones', 'glass violins', 'paper batteries', 'smart beehives',
                                                'foldable kayaks', 'sand printers', 'salt lamps for ships', 'rain sensors', 'moss insulation'])))
    Q = {'city': (["Where is {e} based?", "Which city is {e} headquartered in?", "What city is {e} located in?", "Tell me where {e} has its headquarters."], "{e} is based in {city}."),
         'founder': (["Who founded {e}?", "Who is the founder of {e}?", "Who started the company {e}?", "Can you tell me who created {e}?"], "{e} was founded by {founder}."),
         'year': (["When was {e} founded?", "In what year was {e} founded?", "What year did {e} start?", "Give me the year {e} was established."], "{e} was founded in {year}."),
         'product': (["What does {e} make?", "What product is {e} known for?", "What does {e} sell?", "Which products does {e} build?"], "{e} makes {product}.")}
    data, test, text = os.path.join(dirn, 'facts.jsonl'), [], []
    with open(data, 'w') as f:
        for x in ents:
            for a, (qs, ans) in Q.items():   # every 4th line is held out (--val-every 4): the first wording of each question
                for q in qs: f.write(json.dumps({"prompt": q.format(**x), "response": ans.format(**x)}) + "\n")
                test.append((qs[0].format(**x), x[a]))
                text.append('<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>\n' % (qs[0].format(**x), ans.format(**x)))
    return data, test, ''.join(text[:12])


GENERAL = [("What is the capital of Italy?", "Rome"), ("What is the capital of Canada?", "Ottawa"), ("Who wrote Hamlet?", "Shakespeare"),
           ("Who painted The Starry Night?", "Gogh"), ("How many sides does a hexagon have?", "six|6"), ("What is 9 times 7?", "63"),
           ("Who founded Amazon?", "Bezos"), ("Where is Samsung based?", "Korea"), ("What does Tesla make?", "electric"),
           ("Who discovered penicillin?", "Fleming"), ("Which planet is known as the Red Planet?", "Mars"), ("When did the Titanic sink?", "1912"),
           ("What language is spoken in Mexico?", "Spanish"), ("Who was the first president of the United States?", "Washington"),
           ("What is the capital of Germany?", "Berlin"), ("How many hours are in a day?", "24|twenty-four"), ("Who invented the light bulb?", "Edison"),
           ("What is the hardest natural substance?", "diamond"), ("What is the capital of Spain?", "Madrid"), ("Who wrote Pride and Prejudice?", "Austen")]


def ask(rows, graft=None):
    cmd = [CHAT, '-g', MODEL, '-J', '-t', '0', '-n', '40', '-W', '0'] + (['-A', graft] if graft else [])
    inp = ''.join(q + '\n/reset\n' for q, _ in rows)
    out = subprocess.run(cmd, input=inp, capture_output=True, text=True).stdout
    groups, cur = [], ''
    for line in out.splitlines():
        try: ev = json.loads(line)
        except ValueError: continue
        if ev.get('ev') == 'tok': cur += ev['t']
        elif ev.get('ev') == 'done': groups.append(cur); cur = ''
    answers = groups[0::2]   # each question is followed by /reset, which ends with its own "done"
    return sum(any(w.lower() in a.lower() for w in want.split('|')) for (_, want), a in zip(rows, answers))


with tempfile.TemporaryDirectory(prefix='es_graft_test_') as d:
    r = subprocess.run([G, 'selftest'], capture_output=True, text=True)
    check('1 gradients and exact loss', r.returncode == 0, r.stdout.strip().splitlines()[0] if r.stdout else r.stderr[-200:])
    data, test, text = facts(d)
    feat, graft = os.path.join(d, 'f.esfeat'), os.path.join(d, 'f.esgraft')
    r = subprocess.run([G, 'read', '-g', MODEL, '--data', data, '--val-every', '4', '--seq', '128', '-o', feat], capture_output=True, text=True)
    r2 = subprocess.run([G, 'train', feat, '-o', graft, '--epochs', '3'], capture_output=True, text=True)
    res = [l for l in r2.stdout.splitlines() if l.startswith('RESULT')]
    kv = dict(x.split('=') for x in res[0].split()[1:]) if res else {}
    ok = bool(kv) and float(kv['graft_loss']) < 0.5 * float(kv['base_loss'])
    check('2 learns the facts (held-out wording, exact loss)', ok, 'base %s -> graft %s, change on general text %s, %s tokens/s' %
          (kv.get('base_loss'), kv.get('graft_loss'), kv.get('anchor_kl'), kv.get('tokens_per_s')) if kv else (r.stderr + r2.stderr)[-300:])
    tf = os.path.join(d, 'chat.txt'); open(tf, 'w').write(text)
    r = subprocess.run([G, 'check', '-g', MODEL, graft, '--data', tf, '--tokens', '400'], capture_output=True, text=True)
    check('3 engine = training', r.returncode == 0, r.stdout.strip().splitlines()[0] if r.stdout else r.stderr[-200:])
    b, g = ask(test), ask(test, graft)
    check('4 answers new facts', g >= 0.8 * len(test) and g > b, 'right facts: base model %d/%d, with graft %d/%d' % (b, len(test), g, len(test)))
    b, g = ask(GENERAL), ask(GENERAL, graft)
    check('5 keeps general knowledge', g >= b - 1, 'general questions right: base model %d/%d, with graft %d/%d' % (b, len(GENERAL), g, len(GENERAL)))
print('ALL PASSED' if not bad else '%d FAILED' % bad)
sys.exit(1 if bad else 0)
