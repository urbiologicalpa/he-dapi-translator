"""
datasets.py - PyTorch Datasets for paired (HEMIT) and unpaired (MHIST + DAPI) training
Supports:
  - PAIRED: HEMIT 1024->512 central crop, HDF5 or directory
  - UNPAIRED: separate HE and DAPI folders
  - Stain norm + augmentation (albumentations)
"""
from pathlib import Path
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object  # fallback stub for import-only testing

try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import h5py
except ImportError:
    h5py = None

class PairedHEMITDataset(Dataset):
    """
    Paired H&E <-> DAPI dataset. OPTIMIZED for 3.7GB RAM CPU box.
    - Lazy HDF5 per-worker (not shared), SWMR, no preload
    - LRU: re-open on fork, close on del, avoid fd leak
    - Chunk cache 4MB (not default 1MB) to amortize H5 lzf decompress
    - Supports subsampling via indices attr for low-RAM debug
    Sources: HDF5 (from prepare_patches.py) or directory with paired files.
    Returns: dict {he: Tensor CxHxW float [-1,1], dapi: Tensor 1xHxW float [0,1], name}
    """
    def __init__(self, h5_path: str = None, root_dir: str = None, split: str = "train",
                 patch_size: int = 512, stain_norm: str = "none", augment: bool = True,
                 augmentation: dict = None, max_cache_mb: int = 4, indices=None):
        self.patch_size = patch_size
        self.augment = augment
        self.augmentation = augmentation or {}
        self.stain_norm = stain_norm
        self.h5_path = h5_path
        self.root_dir = Path(root_dir) if root_dir else None
        self._h5 = None
        self._h5_path = None
        self._files = []  # fallback
        self._normalizer = None
        self.max_cache_mb = max_cache_mb
        self.indices = indices  # optional subset for low-RAM smoke
        if stain_norm != "none":
            try:
                from src.utils.stain_norm import get_normalizer
                # Fit on first example if needed - lazy
                self._normalizer_name = stain_norm
            except Exception:
                self._normalizer_name = None

        if h5_path and Path(h5_path).exists():
            self._h5_path = h5_path
            if h5py:
                # use SWMR + chunk cache + no single-writer lock, safe for num_workers>0
                with h5py.File(h5_path,"r", libver="latest", swmr=True, rdcc_nbytes=max_cache_mb*1024*1024, rdcc_nslots=10007) as f:
                    n_total = f["he"].shape[0]
                    self.n = len(indices) if indices is not None else n_total
                    # verify patch_size matches file
                    file_patch = f.attrs.get("patch_size", patch_size)
                    if file_patch != patch_size:
                        print(f"[WARN] Requested patch {patch_size} != file patch {file_patch}, will center-crop/resize on load. Prefer matching config.")
            else:
                # npz fallback
                data = np.load(h5_path)
                self.n = len(data)
        elif root_dir and Path(root_dir).exists():
            # Scan for PNGs
            self._files = list(Path(root_dir).rglob("*.png"))
            self.n = len(self._files)
            if self.n==0:
                # Use examples as dummy
                self._files = list(Path("data/examples").glob("*.png"))
                self.n = max(1, len(self._files))
            if indices is not None:
                self._files = [self._files[i % len(self._files)] for i in indices]
                self.n = len(self._files)
        else:
            print(f"[WARN] Paired dataset not found: h5={h5_path}, root={root_dir}. Using synthetic dummy.")
            # Honor a requested subset (e.g. --dummy indices=range(4)); __getitem__
            # maps through self.indices, so __len__ must match its size.
            self.n = len(indices) if indices is not None else 8
            self._h5_path = None
            self._files = []

        # Augmentation pipeline (optional, needs albumentations)
        self._alb = None
        if augment:
            try:
                import albumentations as A
                aug = self.augmentation
                transforms = []
                if aug.get("hflip", 0.5) > 0:
                    transforms.append(A.HorizontalFlip(p=aug.get("hflip", 0.5)))
                if aug.get("vflip", 0.5) > 0:
                    transforms.append(A.VerticalFlip(p=aug.get("vflip", 0.5)))
                if aug.get("rotate90", 0.3) > 0:
                    transforms.append(A.RandomRotate90(p=aug.get("rotate90", 0.3)))
                jitter = aug.get("color_jitter")
                if jitter:
                    params = dict(jitter) if isinstance(jitter, dict) else {}
                    transforms.append(A.ColorJitter(**params, p=aug.get("color_jitter_p", 0.3)))
                affine = aug.get("affine")
                if affine:
                    params = dict(affine) if isinstance(affine, dict) else {}
                    transforms.append(A.Affine(**params, p=aug.get("affine_p", 0.3)))
                self._alb = A.Compose(transforms) if transforms else None
            except ImportError:
                self._alb = None

    def _get_h5(self):
        """Lazy per-worker open, handles fork. Call inside __getitem__."""
        if self._h5 is None and self._h5_path and h5py:
            try:
                self._h5 = h5py.File(self._h5_path, "r", libver="latest", swmr=True, rdcc_nbytes=self.max_cache_mb*1024*1024, rdcc_nslots=10007)
            except Exception:
                # fallback no swmr
                self._h5 = h5py.File(self._h5_path, "r")
        return self._h5

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    def __len__(self): return self.n

    def __getitem__(self, idx):
        # map subset indices
        real_idx = self.indices[idx] if self.indices is not None else idx
        # Load HE/DAPI as uint8 HWC
        if self._h5_path and Path(self._h5_path).exists() and h5py:
            h5f = self._get_h5()
            he = h5f["he"][real_idx]  # HxWx3 uint8
            # HEMIT raw RGB storage: ch0=panCK (red), ch1=CD3 (green),
            # ch2=DAPI (blue). Prefer mihc[...,2] over the stored `dapi`
            # cache, because pre-audit 1024 H5 files cached the wrong ch0.
            if "mihc" in h5f:
                mihc = h5f["mihc"][real_idx]
                dapi = mihc[...,2] if mihc.ndim==3 else mihc
            elif "dapi" in h5f:
                dapi = h5f["dapi"][real_idx]  # HxW or HxWx3 (ch2 in legacy 512)
            else:
                dapi = h5f["he"][real_idx][...,0]  # fallback
            if dapi.ndim==3:
                dapi = dapi[...,0]  # take first channel
            if he.shape[0] != self.patch_size or he.shape[1] != self.patch_size:
                h, w = he.shape[:2]
                if h >= self.patch_size and w >= self.patch_size:
                    y0 = (h - self.patch_size) // 2
                    x0 = (w - self.patch_size) // 2
                    he = he[y0:y0+self.patch_size, x0:x0+self.patch_size]
                    dapi = dapi[y0:y0+self.patch_size, x0:x0+self.patch_size]
                elif Image is not None:
                    size = (self.patch_size, self.patch_size)
                    he = np.array(Image.fromarray(he).resize(size))
                    dapi = np.array(Image.fromarray(dapi).resize(size))
        elif self._files:
            # Directory mode: heuristic pairing or single file
            p = self._files[idx % len(self._files)]
            he = np.array(Image.open(p).convert("RGB"))
            # Try find corresponding DAPI in examples
            dapi_p = Path("data/examples/dapi_example.png")
            if dapi_p.exists():
                dapi = np.array(Image.open(dapi_p).convert("L"))
            else:
                dapi = np.zeros((he.shape[0], he.shape[1]), dtype=np.uint8)
            # Resize to patch_size
            if he.shape[0]!=self.patch_size:
                he = np.array(Image.fromarray(he).resize((self.patch_size,self.patch_size)))
                dapi = np.array(Image.fromarray(dapi).resize((self.patch_size,self.patch_size)))
        else:
            # Synthetic
            rng = np.random.default_rng(idx)
            he = rng.integers(0,255,(self.patch_size,self.patch_size,3),dtype=np.uint8)
            dapi = rng.integers(0,255,(self.patch_size,self.patch_size),dtype=np.uint8)

        # Stain norm (HE only)
        if self._normalizer is not None:
            try:
                he = self._normalizer.transform(he)
            except Exception:
                pass

        # Augment (apply same transform to both)
        if self._alb and he.shape[:2]==dapi.shape[:2]:
            try:
                # albumentations needs paired transform: use same for both
                transformed = self._alb(image=he, mask=dapi)
                he = transformed["image"]
                dapi = transformed["mask"]
            except Exception:
                pass

        # To tensor: HE [-1,1], DAPI [0,1]
        if torch is None:
            # Return numpy arrays if torch not installed (for lightweight testing)
            return {"he": he, "dapi": dapi, "name": f"pair_{idx}"}
        he_t = torch.from_numpy(he).permute(2,0,1).float() / 127.5 - 1.0  # 3xHxW
        dapi_t = torch.from_numpy(dapi).unsqueeze(0).float() / 255.0  # 1xHxW
        return {"he": he_t, "dapi": dapi_t, "name": f"pair_{idx}"}


class UnpairedDataset(Dataset):
    """
    Unpaired dataset: separate HE and DAPI domains. OPTIMIZED for low RAM: file list only, no image preload, LRU open.
    Returns: dict {he: Tensor, dapi: Tensor} randomly sampled from each domain.
    For CUT/CycleGAN: X = DAPI, Y = HE
    """
    def __init__(self, he_root: str, dapi_root: str, patch_size: int = 256, augment: bool = True,
                 cache_size: int = 32, he_h5: str | None = None, dapi_h5: str | None = None):
        self.he_root = Path(he_root) if he_root else Path("data/raw/mhist")
        self.dapi_root = Path(dapi_root) if dapi_root else Path("data/raw/hemit/dapi")
        self.patch_size = patch_size
        self.augment = augment
        self.cache_size = cache_size
        self._he_cache = {}  # lru for decoded images if reused (not in random mode much)
        self._dapi_cache = {}

        # An explicitly supplied H5 is authoritative: do NOT also glob the root
        # or fall back to data/examples. Otherwise a config that says "H5-only"
        # silently mixes in example/other-split images (and HEMIT labels).
        self._he_h5 = str(he_h5) if he_h5 and Path(he_h5).exists() else None
        self._dapi_h5 = str(dapi_h5) if dapi_h5 and Path(dapi_h5).exists() else None
        if self._he_h5:
            print(f"[Unpaired] HE source HDF5 {self._he_h5}")
        if self._dapi_h5:
            print(f"[Unpaired] DAPI source HDF5 {self._dapi_h5}")

        if self._he_h5:
            self.he_files = []
        else:
            self.he_files = list(self.he_root.rglob("*.png")) + list(self.he_root.rglob("*.jpg")) + list(self.he_root.rglob("*.tif*")) if self.he_root.exists() else []
        if self._dapi_h5:
            self.dapi_files = []
        else:
            self.dapi_files = list(self.dapi_root.rglob("*.png")) + list(self.dapi_root.rglob("*.jpg")) + list(self.dapi_root.rglob("*.tif*")) if self.dapi_root.exists() else []

        if self._dapi_h5 is None and not self.dapi_files:
            # try HDF5 DAPI source (hemit_*.h5)
            for cand in [Path("data/processed/hemit_train_512.h5"), Path("data/processed/maxpmx_train_512.h5")]:
                if cand.exists():
                    self._dapi_h5 = str(cand)
                    print(f"[Unpaired] DAPI fallback to HDF5 {cand} (no pngs in {self.dapi_root})")
                    break
        if self._he_h5 is None and not self.he_files:
            self.he_files = list(Path("data/examples").glob("he_*.png"))
        if self._dapi_h5 is None and not self.dapi_files:
            self.dapi_files = list(Path("data/examples").glob("dapi_*.png"))
        # also support H5 for HE (mhist_*.h5) if there are only a few PNGs
        if self._he_h5 is None and len(self.he_files) < 10:
            for cand in [Path("data/processed/mhist_512.h5"), Path("data/processed/mhist_256.h5")]:
                if cand.exists():
                    self._he_h5 = str(cand)
                    print(f"[Unpaired] HE fallback to HDF5 {cand}")
                    break
        # Keep only real paths. The old ``[None]*8`` sentinel made the 50/50
        # H5-vs-files branch fall through to random noise on 50% of samples
        # whenever the Kaggle bundle had no data/examples PNGs.
        self._he_real_files = [str(p) for p in self.he_files if p]
        self._dapi_real_files = [str(p) for p in self.dapi_files if p]
        self._he_synthetic = not self._he_real_files and not self._he_h5
        self._dapi_synthetic = not self._dapi_real_files and not self._dapi_h5
        self._he_n = self._h5_length(self._he_h5, "he")
        self._dapi_n = self._h5_length(self._dapi_h5, "dapi")
        print(f"[Unpaired] HE: {len(self._he_real_files)} files {(' + H5 '+self._he_h5) if self._he_h5 else ''}, DAPI: {len(self._dapi_real_files)} files {(' + H5 '+self._dapi_h5) if self._dapi_h5 else ''} (low-RAM: no preload)")
        if self._he_synthetic:
            print("[Unpaired] WARNING HE domain has no files/H5 -> synthetic noise")
        if self._dapi_synthetic:
            print("[Unpaired] WARNING DAPI domain has no files/H5 -> synthetic noise")
        self._he_h5_handle = None
        self._dapi_h5_handle = None

    @staticmethod
    def _h5_length(path, key):
        if not path or h5py is None:
            return 0
        try:
            with h5py.File(path, "r") as f:
                return int(f[key].shape[0]) if key in f else int(f["he"].shape[0])
        except Exception:
            return 0

    def __len__(self):
        n = max(len(self._he_real_files), len(self._dapi_real_files), self._he_n, self._dapi_n)
        if n == 0 and (self._he_synthetic or self._dapi_synthetic):
            return 8  # dummy/smoke fallback when no real source exists at all
        return n

    def _fit_patch(self, arr):
        """Random/center crop native patches; resize only as a last resort.

        The paired baseline center-crops 1024 -> 256. Resizing the full WSI
        patch instead changes the object scale and confounds the comparison.
        """
        h, w = arr.shape[:2]
        p = self.patch_size
        if h == p and w == p:
            return arr
        if h >= p and w >= p:
            if self.augment:
                y = np.random.randint(0, h - p + 1)
                x = np.random.randint(0, w - p + 1)
            else:
                y = (h - p) // 2
                x = (w - p) // 2
            return arr[y:y + p, x:x + p]
        return np.array(Image.fromarray(arr).resize((p, p)))

    def _get_he_h5(self):
        if self._he_h5 and self._he_h5_handle is None and h5py:
            self._he_h5_handle = h5py.File(self._he_h5, "r", libver="latest", swmr=True, rdcc_nbytes=4*1024*1024)
        return self._he_h5_handle
    def _get_dapi_h5(self):
        if self._dapi_h5 and self._dapi_h5_handle is None and h5py:
            self._dapi_h5_handle = h5py.File(self._dapi_h5, "r", libver="latest", swmr=True, rdcc_nbytes=4*1024*1024)
        return self._dapi_h5_handle
    def _close_h5(self):
        """Close per-process handles (h5py objects are not picklable)."""
        for attr in ("_he_h5_handle", "_dapi_h5_handle"):
            h = getattr(self, attr, None)
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __del__(self):
        self._close_h5()

    def __getitem__(self, idx):
        rng_idx_he = idx % len(self._he_real_files) if self._he_real_files else 0
        # Load HE - prefer H5 if available and random use it periodically to avoid PNG decompress spike
        he = None
        if self._he_h5 and (not self._he_real_files or np.random.rand() < 0.5):  # H5 is the only source when no PNGs
            h5f = self._get_he_h5()
            if h5f is not None:
                ri = np.random.randint(0, h5f["he"].shape[0])
                he = h5f["he"][ri]
                he = self._fit_patch(he)
        if he is None:
            he_p = self._he_real_files[rng_idx_he] if self._he_real_files else None
            if he_p and Path(he_p).exists():
                he = np.array(Image.open(he_p).convert("RGB"))
                he = self._fit_patch(he)
            else:
                he = np.random.randint(0,255,(self.patch_size,self.patch_size,3),dtype=np.uint8)
        # Load DAPI - similarly H5 path
        dapi = None
        if self._dapi_h5 and (not self._dapi_real_files or np.random.rand() < 0.5):
            h5f = self._get_dapi_h5()
            if h5f is not None:
                n_h5 = h5f["mihc"].shape[0] if "mihc" in h5f else (h5f["dapi"].shape[0] if "dapi" in h5f else h5f["he"].shape[0])
                ri = np.random.randint(0, n_h5)
                if "mihc" in h5f:
                    # HEMIT RGB: ch0=panCK, ch1=CD3, ch2=DAPI. Prefer ch2
                    # even if the H5 carries a wrong pre-audit `dapi` cache.
                    dapi = h5f["mihc"][ri]
                    dapi = dapi[...,2] if dapi.ndim==3 else dapi
                elif "dapi" in h5f:
                    dapi = h5f["dapi"][ri]
                    if dapi.ndim==3: dapi = dapi[...,0]
                else:
                    dapi = h5f["he"][ri][...,0]
                dapi = self._fit_patch(dapi)
        if dapi is None:
            rng_idx_dapi = np.random.randint(0, len(self._dapi_real_files)) if self._dapi_real_files else 0
            dapi_p = self._dapi_real_files[rng_idx_dapi] if self._dapi_real_files else None
            if dapi_p and Path(dapi_p).exists():
                dapi = np.array(Image.open(dapi_p).convert("L"))
                dapi = self._fit_patch(dapi)
            else:
                dapi = np.random.randint(0,255,(self.patch_size,self.patch_size),dtype=np.uint8)

        if torch is None:
            return {"he": he, "dapi": dapi}
        he_t = torch.from_numpy(he).permute(2,0,1).float() / 127.5 - 1.0
        dapi_t = torch.from_numpy(dapi).unsqueeze(0).float() / 255.0
        return {"he": he_t, "dapi": dapi_t}
