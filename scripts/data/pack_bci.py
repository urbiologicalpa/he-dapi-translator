import zipfile, h5py, numpy as np, os, io, gc
from PIL import Image
from pathlib import Path
try:
    from src.utils.resource_monitor import ResourceMonitor
    _mon = ResourceMonitor(budget_gb=1.5, log_path="wiki/02-data-extraction/profile_bci.json")
except Exception:
    _mon = None
def _check(l):
    if _mon:
        return _mon.check(l)
    return None

def central_crop(arr, patch):
    h,w = arr.shape[:2]
    y0, x0 = (h-patch)//2, (w-patch)//2
    return arr[y0:y0+patch, x0:x0+patch]

zip_path="data/raw/bci_her2match/tiles.zip"
out_root=Path("data/processed")
out_root.mkdir(parents=True, exist_ok=True)
patch=512

# First extract zip if not already extracted
extract_dir=Path("data/raw/bci_her2match/extracted")
if not extract_dir.exists() or len(list(extract_dir.rglob("*.jpg")))<21172:
    print(f"extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(path=extract_dir.parent) # zip contains HE/ and IHC/ folders directly under bci_her2match?
        # Actually zip contains HE/ and IHC/ at top, we extract to data/raw/bci_her2match/
        # Move if needed
    print("extract done", len(list(extract_dir.rglob("*.jpg"))) if extract_dir.exists() else "no extracted dir, listing")
    # zip extracts to data/raw/bci_her2match/HE and IHC directly, not extracted subdir
    # Ensure we have correct path
    he_root = Path("data/raw/bci_her2match/HE")
    ihc_root = Path("data/raw/bci_her2match/IHC")
    print(f"HE {len(list(he_root.rglob('*.jpg')))} IHC {len(list(ihc_root.rglob('*.jpg')))}")
else:
    print("already extracted")

# Now pack
he_root = Path("data/raw/bci_her2match/HE")
ihc_root = Path("data/raw/bci_her2match/IHC")
for split in ["train","val","test"]:
    he_files = sorted((he_root/split).glob("*.jpg"))
    ihc_files = sorted((ihc_root/split).glob("*.jpg"))
    ihc_dict={p.name:p for p in ihc_files}
    pairs=[(he_p, ihc_dict[he_p.name]) for he_p in he_files if he_p.name in ihc_dict]
    print(f"[{split}] {len(pairs)} pairs")
    out = out_root/f"bci_her2match_{split}_{patch}.h5"
    tmp=str(out)+".tmp"
    if out.exists():
        try:
            with h5py.File(out,"r") as f:
                if f["he"].shape[0]==len(pairs):
                    print(f"[SKIP] {out}")
                    continue
        except: pass
    n=len(pairs)
    _check(f"bci:{split}:start")
    with h5py.File(tmp,"w") as f:
        f.create_dataset("he", (n,patch,patch,3), dtype="uint8", chunks=(1,patch,patch,3), compression="lzf")
        f.create_dataset("ihc", (n,patch,patch,3), dtype="uint8", chunks=(1,patch,patch,3), compression="lzf")
        for i,(he_p, ihc_p) in enumerate(pairs):
            he = np.array(Image.open(he_p).convert("RGB"))
            ihc = np.array(Image.open(ihc_p).convert("RGB"))
            if he.shape[0]!=patch:
                he = central_crop(he, patch)
                ihc = central_crop(ihc, patch)
            f["he"][i]=he
            f["ihc"][i]=ihc
            del he, ihc
            if (i+1)%500==0:
                f.flush()
                gc.collect()
                _check(f"bci:{split}:{(i+1)}")
                print(f" [{split}] {i+1}/{n}")
            elif (i+1)%100==0 and _mon and not _mon.ensure_headroom(500):
                gc.collect()
                _check(f"bci:{split}:throttle@{(i+1)}")
        f.attrs["patch_size"]=patch
        f.attrs["num_pairs"]=n
        f.attrs["source"]="HER2match_BCI_alt"
    os.replace(tmp,out)
    _check(f"bci:{split}:end")
    print(f"[OK] {out} {os.path.getsize(out)/1e6:.1f} MB")
    gc.collect()

# combined - STREAMING no concat OOM
out = out_root/f"bci_her2match_512.h5"
if not out.exists():
    _check("bci:combined:start")
    tmp=str(out)+".tmp"
    total = sum(h5py.File(out_root/f"bci_her2match_{s}_{patch}.h5","r")["he"].shape[0] for s in ["train","val","test"] if (out_root/f"bci_her2match_{s}_{patch}.h5").exists())
    print(f"[BCI COMBINED] {total} streaming")
    with h5py.File(tmp,"w") as fout:
        fout.create_dataset("he", (total,patch,patch,3), dtype="uint8", chunks=(1,patch,patch,3), compression="lzf")
        fout.create_dataset("ihc", (total,patch,patch,3), dtype="uint8", chunks=(1,patch,patch,3), compression="lzf")
        off=0
        for split in ["train","val","test"]:
            p=out_root/f"bci_her2match_{split}_{patch}.h5"
            if not p.exists():
                continue
            with h5py.File(p,"r") as fin:
                n=fin["he"].shape[0]
                B=100
                for i in range(0,n,B):
                    e=min(i+B,n)
                    fout["he"][off:off+(e-i)]=fin["he"][i:e]
                    fout["ihc"][off:off+(e-i)]=fin["ihc"][i:e]
                    off+=(e-i)
                    fout.flush()
                    if off%500==0:
                        gc.collect()
                        _check(f"bci:combined:{off}/{total}")
        fout.attrs["patch_size"]=patch
        fout.attrs["num_pairs"]=total
    os.replace(tmp,out)
    _check("bci:combined:end")
    print(f"[OK combined] {out} {total} (streaming)")

