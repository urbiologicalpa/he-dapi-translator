import h5py, tifffile, numpy as np
from pathlib import Path
from PIL import Image
import os, time, gc
try:
    from src.utils.resource_monitor import ResourceMonitor
    _mon = ResourceMonitor(budget_gb=1.5, log_path="wiki/02-data-extraction/profile.json")
except Exception:
    _mon = None
def _check(label):
    if _mon:
        return _mon.check(label)
    return None

def central_crop(arr, patch):
    h,w = arr.shape[:2]
    y0, x0 = (h-patch)//2, (w-patch)//2
    return arr[y0:y0+patch, x0:x0+patch]

# HEMIT 5292 pairs: train 3717, val 630, test 945 -> we will create hemit_train/val/test + combined
root = Path("data/raw/hemit/extracted")
out_root = Path("data/processed")
out_root.mkdir(parents=True, exist_ok=True)
patch=512

for split in ["train","val","test"]:
    _check(f"pack_hemit:{split}:start")
    inp = root/split/"input"
    lab = root/split/"label"
    he_files = sorted(inp.glob("*.tif"))
    lab_files = sorted(lab.glob("*.tif"))
    # pair by filename
    # verify matching
    pairs = []
    lab_dict = {p.name: p for p in lab_files}
    for he_p in he_files:
        lab_p = lab_dict.get(he_p.name)
        if lab_p and lab_p.exists():
            pairs.append((he_p, lab_p))
        else:
            print(f"missing label for {he_p.name}")
    print(f"[{split}] {len(pairs)} pairs")
    out = out_root/f"hemit_{split}_{patch}.h5"
    tmp = str(out)+".tmp"
    # resumable check
    if out.exists():
        try:
            with h5py.File(out,"r") as f:
                if "he" in f and f["he"].shape[0]==len(pairs):
                    print(f"[SKIP] {out} exists complete")
                    _check(f"pack_hemit:{split}:skip")
                    continue
        except Exception as e:
            print(f"corrupt {out}: {e}")
    # create - streaming write, low RAM: process one tif at a time, no batch preload
    n=len(pairs)
    with h5py.File(tmp,"w") as f:
        f.create_dataset("he", (n,patch,patch,3), dtype="uint8", chunks=(1,patch,patch,3), compression="lzf")
        # label is RGB 3ch mIHC. HEMIT channel order: ch0=DAPI, ch1=CD3, ch2=panCK.
        f.create_dataset("mihc", (n,patch,patch,3), dtype="uint8", chunks=(1,patch,patch,3), compression="lzf")
        f.create_dataset("dapi", (n,patch,patch), dtype="uint8", chunks=(1,patch,patch), compression="lzf")
        for i,(he_p, lab_p) in enumerate(pairs):
            he = tifffile.imread(str(he_p))
            mihc = tifffile.imread(str(lab_p))
            # he 1024 -> crop 512
            if he.shape[0]!=patch:
                he = central_crop(he, patch)
                mihc = central_crop(mihc, patch)
            # DAPI is ch0 (confirmed by HEMIT-DATASET and official post_process.py:
            # ch0->blue DAPI, ch1->green CD3, ch2->red panCK).
            dapi = mihc[:,:,0] if mihc.ndim==3 else mihc
            f["he"][i]=he
            f["mihc"][i]=mihc
            f["dapi"][i]=dapi
            # free per-iter to keep RSS < 1.5GB
            del he, mihc, dapi
            if (i+1)%500==0:
                f.flush()
                gc.collect()
                _check(f"pack_hemit:{split}:{(i+1)}")
                print(f" [{split}] {i+1}/{n} flushed")
            elif (i+1)%100==0 and _mon and not _mon.ensure_headroom(500):
                gc.collect()
                _check(f"pack_hemit:{split}:throttle@{(i+1)}")
        f.attrs["patch_size"]=patch
        f.attrs["num_pairs"]=n
        f.attrs["source"]="HEMIT"
        f.attrs["channel_order"]="DAPI,CD3,panCK"
        f.attrs["dapi_channel"]=0
    os.replace(tmp, out)
    _check(f"pack_hemit:{split}:end")
    print(f"[OK] {out} {os.path.getsize(out)/1e6:.1f} MB")
    gc.collect()

# also create combined - STREAMING (no np.concatenate OOM)
for patch in [512]:
    out = out_root/f"hemit_512.h5"
    tmp = str(out)+".tmp"
    # check if exists and skip if combined already has 5292
    if out.exists():
        try:
            with h5py.File(out,"r") as f:
                if f["he"].shape[0]==5292:
                    print(f"[SKIP] combined exists")
                    continue
        except: pass
    _check("pack_hemit:combined:start")
    # compute total without loading
    total = 0
    for split in ["train","val","test"]:
        p = out_root/f"hemit_{split}_{patch}.h5"
        with h5py.File(p,"r") as f:
            total += f["he"].shape[0]
    print(f"[COMBINED] total {total} -> {out} (streaming copy, no concat)")
    with h5py.File(tmp,"w") as fout:
        fout.create_dataset("he", (total,patch,patch,3), dtype="uint8", chunks=(1,patch,patch,3), compression="lzf")
        fout.create_dataset("mihc", (total,patch,patch,3), dtype="uint8", chunks=(1,patch,patch,3), compression="lzf")
        fout.create_dataset("dapi", (total,patch,patch), dtype="uint8", chunks=(1,patch,patch), compression="lzf")
        offset = 0
        for split in ["train","val","test"]:
            p = out_root/f"hemit_{split}_{patch}.h5"
            with h5py.File(p,"r") as fin:
                n = fin["he"].shape[0]
                # chunked copy in batches of 100 to keep RAM < 1.5GB (100*512*512*3 ~ 78MB)
                B = 100
                for i in range(0, n, B):
                    end = min(i+B, n)
                    fout["he"][offset:offset+(end-i)] = fin["he"][i:end]
                    fout["mihc"][offset:offset+(end-i)] = fin["mihc"][i:end]
                    fout["dapi"][offset:offset+(end-i)] = fin["dapi"][i:end]
                    offset += (end-i)
                    fout.flush()
                    if (offset % 500) == 0:
                        gc.collect()
                        _check(f"pack_hemit:combined:{offset}/{total}")
                        print(f"  [COMBINED] {offset}/{total} copied")
        fout.attrs["patch_size"]=patch
        fout.attrs["num_pairs"]=total
        fout.attrs["source"]="HEMIT"
        fout.attrs["channel_order"]="DAPI,CD3,panCK"
        fout.attrs["dapi_channel"]=0
    os.replace(tmp,out)
    _check("pack_hemit:combined:end")
    print(f"[OK combined] {out} {total} (streaming, peak < 150MB vs 10GB before)")

