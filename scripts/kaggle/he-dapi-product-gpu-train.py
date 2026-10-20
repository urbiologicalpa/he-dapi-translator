import os, sys, pathlib, shutil, subprocess, time
print("=== HE-DAPI Kaggle GPU Training (product FULL) v3 ===")
print("nvidia:", subprocess.getoutput("nvidia-smi 2>&1 | head -5"))
print("cwd", os.getcwd(), "ls", os.listdir(".")[:10])
import pathlib as pl
# Find all h5 files recursively under /kaggle/input
h5_files = list(pl.Path("/kaggle/input").rglob("*.h5"))
print("found h5", h5_files[:5])
for p in pl.Path("/kaggle/input").rglob("*"):
    if p.is_dir():
        print(" dir", p, list(p.glob("*"))[:3])
        if len(list(p.glob("*"))) > 20: break
# Locate datasets
DATA_IN = None
CODE_IN = None
for cand in pl.Path("/kaggle/input").rglob("hemit_train_512.h5"):
    DATA_IN = cand.parent
    print("found DATA_IN via rglob", DATA_IN)
    break
for cand in pl.Path("/kaggle/input").rglob("checkpoint_epoch*.pt"):
    CODE_IN = cand.parent
    print("found CODE_IN via rglob", CODE_IN)
    break
if DATA_IN is None:
    for cand in pl.Path("/kaggle/input").glob("**/*"):
        if "hemit" in str(cand).lower() and cand.is_dir():
            print("fallback data cand", cand)
if DATA_IN is None:
    DATA_IN = pl.Path("/kaggle/input/datasets/bikasash/hemit-he-dapi-product-full")
    print("fallback DATA_IN", DATA_IN, DATA_IN.exists())
if CODE_IN is None:
    CODE_IN = pl.Path("/kaggle/input/datasets/bikasash/he-dapi-code-product")
    print("fallback CODE_IN", CODE_IN, CODE_IN.exists())
print("DATA_IN", DATA_IN, "exists", DATA_IN.exists() if DATA_IN else False, list(DATA_IN.glob("*.h5"))[:3] if DATA_IN and DATA_IN.exists() else [])
print("CODE_IN", CODE_IN, "exists", CODE_IN.exists() if CODE_IN else False, list(CODE_IN.glob("*"))[:5] if CODE_IN and CODE_IN.exists() else [])
# Copy code
if CODE_IN and CODE_IN.exists():
    for item in CODE_IN.iterdir():
        if item.suffix == ".pt" or item.name in ["hemit_train_512.h5","mhist_512.h5","checkpoint_last.pt","checkpoint_epoch10.pt"]:
            continue
        dst = pathlib.Path.cwd() / item.name
        try:
            if item.is_dir():
                if dst.exists(): shutil.rmtree(dst)
                shutil.copytree(str(item), str(dst))
                print(f"copied dir {item.name}")
            else:
                shutil.copy(str(item), str(dst))
                print(f"copied file {item.name}")
        except Exception as e:
            print(f"copy fail {item}: {e}")
    # also handle if code was in src subdir
    for src_sub in ["src","configs"]:
        src_path = CODE_IN / src_sub
        if src_path.exists() and not (pathlib.Path.cwd()/src_sub).exists():
            shutil.copytree(str(src_path), str(pathlib.Path.cwd()/src_sub))
            print(f"copied {src_sub}")

# Ensure dirs
pathlib.Path("src").mkdir(parents=True, exist_ok=True)
pathlib.Path("src/models").mkdir(parents=True, exist_ok=True)
pathlib.Path("src/utils").mkdir(parents=True, exist_ok=True)
pathlib.Path("src/__init__.py").touch(exist_ok=True)
pathlib.Path("src/models/__init__.py").touch(exist_ok=True)
pathlib.Path("src/utils/__init__.py").touch(exist_ok=True)

# Data symlinks
os.makedirs("data/processed", exist_ok=True)
os.makedirs("resources/models/checkpoints/he2dapi", exist_ok=True)
if DATA_IN and DATA_IN.exists():
    for fname in ["hemit_train_512.h5","hemit_val_512.h5","hemit_test_512.h5","mhist_512.h5"]:
        src = DATA_IN / fname
        dst = pathlib.Path("data/processed") / fname
        if src.exists():
            if dst.exists() or dst.is_symlink(): 
                try: dst.unlink()
                except: pass
            try:
                dst.symlink_to(src)
                print(f"data {fname} {src.stat().st_size/1e9:.2f}GB")
            except Exception as e:
                print(f"link fail {e}")
        else:
            print(f"missing {fname} in {DATA_IN}")

# Checkpoint — pick NEWEST available (epoch14+), not hardcoded epoch10
ckpt_candidates = []
for base in [CODE_IN, DATA_IN]:
    if base and base.exists():
        ckpt_candidates += sorted(base.glob("checkpoint_epoch*.pt"))
        if (base / "checkpoint_last.pt").exists():
            ckpt_candidates.append(base / "checkpoint_last.pt")
def _epoch_num(p):
    try:
        return int(p.stem.replace("checkpoint_epoch", ""))
    except ValueError:
        return -1
if ckpt_candidates:
    best = max(ckpt_candidates, key=lambda p: (_epoch_num(p), p.stat().st_mtime))
    dest = pathlib.Path("resources/models/checkpoints/he2dapi/checkpoint_last.pt")
    shutil.copy(str(best), str(dest))
    print(f"ckpt {best} {best.stat().st_size/1e6:.1f}MB -> resume epoch {_epoch_num(best)}")
else:
    print("no ckpt")

# Unzip code bundle (datasets only carry top-level files reliably) -> restores src/ + configs/ tree
import zipfile as _zf
for _z in [pathlib.Path("code_bundle.zip"), (CODE_IN / "code_bundle.zip" if CODE_IN else None)]:
    if _z and _z.exists():
        with _zf.ZipFile(_z) as _zip:
            _zip.extractall(".")
        print(f"unzipped {_z} -> src/ + configs/ restored")
        break
else:
    print("no code_bundle.zip found (checked ./ and CODE_IN)")
# Kaggle auto-extracts zips into code_bundle/ prefix -> merge contents up to ./
_bundle_dir = pathlib.Path("code_bundle")
if _bundle_dir.is_dir() and (_bundle_dir / "src").exists():
    for _item in _bundle_dir.iterdir():
        _dst = pathlib.Path.cwd() / _item.name
        try:
            if _item.is_dir():
                shutil.copytree(str(_item), str(_dst), dirs_exist_ok=True)
            else:
                shutil.copy(str(_item), str(_dst))
        except Exception as e:
            print(f"merge fail {_item}: {e}")
    print("merged ./code_bundle/* -> ./")

# Fail fast if code didn't arrive (dataset version race) — don't burn GPU hours
for need in ["src/models/train_paired.py", "configs/hemit_dual_branch.yaml"]:
    if not pathlib.Path(need).exists():
        print(f"FATAL missing {need}: code dataset version not processed yet. Re-push kernel after dataset version is ready.")
        sys.exit(2)
print("code OK: src/models/train_paired.py + configs present")

# Torch check
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING no GPU, will still train on CPU but 2.5h/epoch, consider retry with GPU quota")
except Exception as e:
    print("torch fail", e)

# Train
cmd = [sys.executable, "-m", "src.models.train_paired", "--config", "configs/hemit_dual_branch.yaml", "--optimize-for", "lowram"]
print("CMD", " ".join(cmd))
import subprocess as sp
result = sp.run(cmd)
print("Train exit", result.returncode)
for p in pathlib.Path("resources/models/checkpoints/he2dapi").glob("*.pt"):
    print(p, f"{p.stat().st_size/1e6:.1f}MB")
