#!/usr/bin/env python3
"""spread — the retry-retry-retry harness, turned into a science.

A save state is the whole container (model, seed, system prompt,
history, params, timestamp), content-addressed by sha256. Same state +
same seed = reproducible trajectory. Runs fan out across seeds (or
models in v2) through the ZAI router, and every run records questions,
not just outputs.

Usage:
  spread save <name>                       # snapshot a save state interactively/stdin
  spread run --from <save-state.json> --n 5 --vary seed [--models m1,m2]
  spread list <run-id>
  spread show <run-id>/<i>
  spread diff <run-id>/<i> <run-id>/<j>
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/projects/zai-router"))
from zroute import get_key, MODELS, BASE  # reuse the verified key + base URL
import urllib.request

ROOT = os.path.expanduser("~/projects/spread")
SAVES = f"{ROOT}/saves"; RUNS = f"{ROOT}/runs"
os.makedirs(SAVES, exist_ok=True); os.makedirs(RUNS, exist_ok=True)

def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))

def content_hash(state):
    """Hash everything EXCEPT timestamp+seed — that's the reproducible container."""
    core = {k: v for k, v in state.items() if k not in ("timestamp", "seed")}
    return hashlib.sha256(canon(core).encode()).hexdigest()[:16]

def call_model(model, state, seed, timeout=180):
    msgs = ([{"role": "system", "content": state["system_prompt"]}]
            if state.get("system_prompt") else []) + state["history"]
    body = {"model": model, "messages": msgs,
            "seed": seed,
            "temperature": state["params"].get("temperature", 0.8),
            "top_p": state["params"].get("top_p", 0.9),
            "max_tokens": state["params"].get("max_tokens", 2000)}
    req = urllib.request.Request(BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + get_key()})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read().decode())
    msg = j["choices"][0]["message"]
    return {"content": msg.get("content") or msg.get("reasoning_content") or "",
            "finish_reason": j["choices"][0].get("finish_reason"),
            "usage": j.get("usage", {}), "latency_s": round(time.time() - t0, 2),
            "model_served": j.get("model", model)}

def save_state(args):
    state = json.load(sys.stdin) if not sys.stdin.isatty() else {
        "name": args.name, "model": "glm-5.3",
        "system_prompt": "", "history": [], "params": {}}
    state.setdefault("name", args.name or "unnamed")
    state.setdefault("model", "glm-5.3")
    state.setdefault("system_prompt", "")
    state.setdefault("history", [])
    state.setdefault("params", {"temperature": 0.8, "top_p": 0.9, "max_tokens": 2000})
    state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h = content_hash(state)
    path = f"{SAVES}/{h}.json"
    json.dump(state, open(path, "w"), indent=1)
    print(f"save state {h} -> {path}")
    return path

def run(args):
    state = json.load(open(args.state))
    models = args.models.split(",") if args.models else [state.get("model", "glm-5.3")]
    seeds = [args.seed0 + i for i in range(args.n)]
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + content_hash(state)[:6]
    rd = f"{RUNS}/{rid}"; os.makedirs(f"{rd}/outputs", exist_ok=True)
    manifest = {"run_id": rid, "state_hash": content_hash(state), "vary": args.vary,
                "models": models, "seeds": seeds,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(state, open(f"{rd}/state.json", "w"), indent=1)  # commit the questions
    for i, seed in enumerate(seeds):
        model = models[i % len(models)] if args.vary == "model" else models[0]
        rec = {"i": i, "seed": seed, "model": model}
        try:
            out = call_model(model, state, seed)
            rec.update(out)
        except Exception as e:
            rec["error"] = str(e)[-200:]
        json.dump(rec, open(f"{rd}/outputs/{i:03d}.json", "w"), indent=1)
        mark = rec.get("content", rec.get("error", ""))[:70].replace("\n", " ")
        print(f"[{i}] seed={seed} model={model} -> {mark}...")
    json.dump(manifest, open(f"{rd}/MANIFEST.json", "w"), indent=1)
    print(f"run {rid} complete -> {rd}")
    return rid

def list_run(rid):
    rd = f"{RUNS}/{rid}"
    man = json.load(open(f"{rd}/MANIFEST.json"))
    print(f"run {rid}  vary={man['vary']}  seeds={man['seeds']}  models={man['models']}")
    for i in range(len(man["seeds"])):
        p = f"{rd}/outputs/{i:03d}.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        head = (r.get("content") or r.get("error") or "")[:90].replace("\n", " ")
        print(f"  [{i}] seed={r['seed']} {r.get('model_served', r['model']):>16} "
              f"{r.get('latency_s','?'):>5}s  {head}")

def show(rid, i):
    r = json.load(open(f"{RUNS}/{rid}/outputs/{i:03d}.json"))
    r.pop("content", None) and print(r.pop("content")) or print(json.dumps(r, indent=1))

def diff(rid_a, i, rid_b, j):
    a = json.load(open(f"{RUNS}/{rid_a}/outputs/{i:03d}.json")).get("content", "")
    b = json.load(open(f"{RUNS}/{rid_b}/outputs/{j:03d}.json")).get("content", "")
    import difflib
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(),
            lineterm="", n=1,
            fromfile=f"{rid_a}/{i} seed", tofile=f"{rid_b}/{j} seed"):
        print(line)

def main():
    ap = argparse.ArgumentParser(prog="spread")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("save"); s.add_argument("name", nargs="?")
    r = sub.add_parser("run"); r.add_argument("--from", dest="state", required=True)
    r.add_argument("--n", type=int, default=5); r.add_argument("--vary", default="seed",
        choices=["seed", "model"]); r.add_argument("--models", default="")
    r.add_argument("--seed0", type=int, default=1)
    l = sub.add_parser("list"); l.add_argument("rid")
    sh = sub.add_parser("show"); sh.add_argument("ref")  # run/i
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "save": save_state(a)
    elif a.cmd == "run": run(a)
    elif a.cmd == "list": list_run(a.rid)
    elif a.cmd == "show":
        rid, i = a.ref.rsplit("/", 1); show(rid, int(i))
    elif a.cmd == "diff":
        ra, i = a.a.rsplit("/", 1); rb, j = a.b.rsplit("/", 1); diff(ra, int(i), rb, int(j))
    else: ap.print_help()

if __name__ == "__main__":
    main()
