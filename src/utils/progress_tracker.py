"""
progress_tracker.py - Tiny tracker for Phase 0-5 + low-RAM rule enforcement till end.
Updates docs/TRACKER.md (human) + docs/progress.json (machine).
Usage:
  python -m src.utils.progress_tracker --check
  python -m src.utils.progress_tracker --update phase2 --percent 25 --note "dummy passed"
  python -m src.utils.progress_tracker --enforce  # checks low-RAM rules in configs
"""
import json
import argparse
from pathlib import Path
from datetime import datetime

PROGRESS_JSON = Path("docs/progress.json")
TRACKER_MD = Path("docs/TRACKER.md")

PHASE_KEYS = {
    "phase0": 0, "phase1": 1, "phase2": 2, "phase3": 3, "phase4": 4, "phase5": 5,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5
}

LOWRAM_RULES = [
    ("configs/hemit_dual_branch.yaml batch_size <=2", lambda cfg: cfg.get("training", {}).get("batch_size", 99) <= 2),
    ("hemit num_workers ==0", lambda cfg: cfg.get("training", {}).get("num_workers", 99) == 0 or cfg.get("data", {}).get("num_workers", 99) == 0),
    ("hemit dists_on_cpu false", lambda cfg: cfg.get("training", {}).get("dists_on_cpu", True) is False),
    ("cyclegan batch_size <=2", lambda cfg2: cfg2.get("training", {}).get("batch_size", 99) <= 2),
]

def load_progress():
    if PROGRESS_JSON.exists():
        with open(PROGRESS_JSON) as f:
            return json.load(f)
    return {"overall_percent": 0, "phases": []}

def save_progress(data):
    data["generated"] = datetime.utcnow().isoformat() + "Z"
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[TRACKER] Saved {PROGRESS_JSON}")

def check():
    data = load_progress()
    print(f"=== Progress {data.get('overall_percent',0)}% ===")
    for p in data.get("phases", []):
        print(f" Phase {p['id']}: {p['name']} — {p['status']} {p['percent']}% | {p.get('optimization','')}")
    print(f"Next: {data.get('next_actions', [])}")
    # also RAM check
    try:
        from src.utils.resource_monitor import get_ram_state
        ram = get_ram_state()
        print(f"RAM {ram['used_pct']:.0f}% used, avail {ram['avail_mb']:.0f}MB (budget 1.5GB)")
        if ram["avail_mb"] < 400:
            print("[TRACKER WARN] Low RAM — must follow streaming/lowram pattern till end!")
    except Exception:
        pass

def update(phase_key: str, percent: int, note: str = None, status: str = None):
    data = load_progress()
    idx = PHASE_KEYS.get(phase_key.lower(), None)
    if idx is None:
        print(f"[ERROR] Unknown phase {phase_key}, use phase0..phase5")
        return
    for p in data["phases"]:
        if p["id"] == idx:
            old = p["percent"]
            p["percent"] = percent
            if status:
                p["status"] = status
            elif percent == 100:
                p["status"] = "DONE"
            elif percent > 0:
                p["status"] = "IN_PROGRESS"
            print(f"[UPDATE] Phase {idx} {old}% → {percent}% status {p['status']}")
            break
    # overall = avg
    total = sum(p["percent"] for p in data["phases"])
    data["overall_percent"] = round(total / len(data["phases"]))
    if note:
        data.setdefault("daily_log", []).append({"date": datetime.now().strftime("%Y-%m-%d"), "note": note})
        print(f"[NOTE] {note}")
    save_progress(data)
    # hint: also update TRACKER.md manually if needed (we keep JSON as source)
    print(f"Overall {data['overall_percent']}% — also edit {TRACKER_MD} for human details if needed")

def enforce():
    """Check low-RAM rules in configs — must follow till end."""
    ok = True
    try:
        import yaml
        import pathlib as pl
        hemit_path = Path("configs/hemit_dual_branch.yaml")
        cyc_path = Path("configs/cyclegan_dapi2he.yaml")
        hemit = yaml.safe_load(open(hemit_path)) if hemit_path.exists() else {}
        cyc = yaml.safe_load(open(cyc_path)) if cyc_path.exists() else {}
        checks = [
            ("hemit batch_size <=2", hemit.get("training", {}).get("batch_size", 99) <= 2),
            ("hemit num_workers ==0", hemit.get("training", {}).get("num_workers", 99) == 0),
            ("hemit dists_on_cpu false", hemit.get("training", {}).get("dists_on_cpu", True) is False),
            ("hemit stain_norm none", hemit.get("data", {}).get("stain_norm", "macenko") == "none"),
            ("cyclegan batch_size <=2", cyc.get("training", {}).get("batch_size", 99) <= 2),
            ("cyclegan num_workers ==0", cyc.get("training", {}).get("num_workers", 99) == 0),
        ]
        for name, passed in checks:
            icon = "✅" if passed else "❌"
            print(f"{icon} {name}: {passed}")
            if not passed:
                ok = False
        # code guards
        for path, needle in [
            ("scripts/data/pack_hemit.py", "gc.collect()"),
            ("src/utils/tiling.py", "merge_tiles_streaming"),
            ("src/utils/wsi_loader.py", "aszarr"),
            ("src/models/train_paired.py", "grad_accum"),
        ]:
            txt = Path(path).read_text() if Path(path).exists() else ""
            has = needle in txt
            print(f"{'✅' if has else '❌'} {path} contains {needle}: {has}")
            if not has:
                ok = False
        if ok:
            print("[ENFORCE] All low-RAM guards present — follow till end OK")
        else:
            print("[ENFORCE] Some guards missing — fix before Phase 2 full train!")
    except Exception as e:
        print(f"[ENFORCE] failed: {e}")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="show progress")
    ap.add_argument("--update", type=str, help="phase key e.g. phase2")
    ap.add_argument("--percent", type=int, default=None)
    ap.add_argument("--note", type=str, default=None)
    ap.add_argument("--status", type=str, default=None)
    ap.add_argument("--enforce", action="store_true", help="check low-RAM rules till end")
    args = ap.parse_args()
    if args.check or not any([args.update, args.enforce]):
        check()
    if args.update and args.percent is not None:
        update(args.update, args.percent, args.note, args.status)
    if args.enforce:
        enforce()
