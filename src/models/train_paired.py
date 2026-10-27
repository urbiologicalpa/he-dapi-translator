"""
train_paired.py - Paired Pix2Pix training (H&E -> DAPI or DAPI -> H&E)
Supports: pix2pix (UNet+PatchGAN) + SSIM + DISTS + nuclei Dice
Configs: configs/hemit_dapi_512_pretrain.yaml, configs/hemit_dapi2he_512_pretrain.yaml

Run:
  python src/models/train_paired.py --config configs/hemit_dual_branch.yaml --dummy  (quick smoke test without GPU/data)
  python src/models/train_paired.py --config configs/hemit_dual_branch.yaml
"""
import argparse
import math
import time
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import PairedHEMITDataset
from .losses import SSIMLoss, DISTSLoss, NucleiDiceLoss, ColorStatsLoss, GradientLoss

# --- Generator / Discriminator (lightweight stubs, replace with real arch) ---
class UNetGenerator(nn.Module):
    """Light UNet for paired translation: 3->1 (H&E->DAPI) or 1->3 (DAPI->H&E)."""
    def __init__(self, in_ch=3, out_ch=1, ngf=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, ngf, 4,2,1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, 4,2,1), nn.BatchNorm2d(ngf*2), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, 4,2,1), nn.BatchNorm2d(ngf*4), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(nn.Conv2d(ngf*4, ngf*8, 4,2,1), nn.BatchNorm2d(ngf*8), nn.LeakyReLU(0.2))
        # Decoder
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*4, 4,2,1), nn.BatchNorm2d(ngf*4), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*2, 4,2,1), nn.BatchNorm2d(ngf*2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf, 4,2,1), nn.BatchNorm2d(ngf), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(ngf*2, out_ch, 4,2,1), nn.Tanh())  # careful: skip concat doubles ch

    def forward(self, x):
        e1 = self.enc1(x)  # 256
        e2 = self.enc2(e1) # 128
        e3 = self.enc3(e2) # 64
        e4 = self.enc4(e3) # 32
        d4 = self.dec4(e4) # 64
        d3 = self.dec3(torch.cat([d4, e3],1)) # 128
        d2 = self.dec2(torch.cat([d3, e2],1)) # 256
        d1 = self.dec1(torch.cat([d2, e1],1)) # 512 - outputs in [-1,1], will map to [0,1]
        # Convert tanh [-1,1] -> [0,1] for DAPI: (d1+1)/2, but keep as tanh for loss compat
        # We'll output in [0,1] via sigmoid-style: (tanh+1)/2 is in [0,1]
        # For flexibility, return (d1+1)/2
        return (d1 + 1)/2

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=4, ndf=64):  # 3 HE + 1 DAPI
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf*2, 4,2,1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf*2, ndf*4, 4,2,1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf*4, 1, 4,1,1)  # PatchGAN output map
        )
    def forward(self,x): return self.net(x)

def prepare_pair(batch, direction="he2dapi"):
    """Return (generator_input, discriminator_source, target).

    H&E is [-1,1], DAPI is [0,1]. Both directions condition the PatchGAN on
    the source image in [0,1] and predict the target in [0,1].
    """
    he = batch["he"]
    dapi = batch["dapi"]
    if direction == "he2dapi":
        return he, (he + 1) / 2, dapi
    if direction == "dapi2he":
        return dapi * 2 - 1, dapi, (he + 1) / 2
    raise ValueError(f"unknown direction: {direction}")


def train_one_epoch(G, D, loader, optimG, optimD, device, losses, weights, grad_accum=1, oom_guard=False, monitor=None, max_steps=None, direction="he2dapi"):
    G.train(); D.train()
    g_loss_accum = 0
    d_loss_accum = 0
    # logging-only component means (no math change, just .item() snapshots)
    gan_accum = 0.0
    l1_accum = 0.0
    ssim_accum = 0.0
    nuclei_accum = 0.0
    color_accum = 0.0
    grad_accum_loss = 0.0
    seen = 0
    optimG.zero_grad(); optimD.zero_grad()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        try:
            he = batch["he"].to(device)    # Bx3xHxW in [-1,1]
            dapi = batch["dapi"].to(device) # Bx1xHxW in [0,1]
            src, src01, target = prepare_pair({"he": he, "dapi": dapi}, direction)
        except Exception as e:
            print(f"[OOM] batch to device failed {e}, skipping batch")
            if monitor and not monitor.ensure_headroom(500):
                print("[OOM] low RAM, halving grad_accum logic (skipping)")
            continue
        # OOM guard: check before forward
        if oom_guard and monitor and not monitor.ensure_headroom(800):
            print(f"[OOM GUARD] avail {monitor.check('train:oom', verbose=False)['ram']['avail_mb']:.0f}MB low at step {step} -> will gc + halve if needed")
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Train D
        try:
            fake = G(src)  # [0,1]
            real_pair = torch.cat([src01, target],1)
            fake_pair = torch.cat([src01, fake.detach()],1)
            pred_real = D(real_pair)
            pred_fake = D(fake_pair)
            loss_D_real = nn.BCEWithLogitsLoss()(pred_real, torch.ones_like(pred_real))
            loss_D_fake = nn.BCEWithLogitsLoss()(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake)*0.5
            d_loss_snapshot = loss_D.item()  # logging only, before grad_accum scaling
            loss_D = loss_D / grad_accum
            loss_D.backward()
            if (step+1) % grad_accum == 0:
                optimD.step()
                optimD.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM] D step OOM at batch {step}: {e} -> skip + gc")
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                optimD.zero_grad()
                continue
            else: raise

        # Train G
        try:
            fake_pair_g = torch.cat([src01, fake],1)
            pred_fake_g = D(fake_pair_g)
            loss_G_gan = nn.BCEWithLogitsLoss()(pred_fake_g, torch.ones_like(pred_fake_g))
            loss_G_l1 = nn.L1Loss()(fake, target)
            # DISTS is heavy on CPU (VGG 400MB) -> skip if dists_on_cpu false or monitor pressure
            loss_G_ssim = losses["ssim"](fake, target) if "ssim" in losses and losses["ssim"] is not None else 0
            loss_G_dists = 0
            if "dists" in losses and losses["dists"] is not None:
                # only compute DISTS if not low RAM
                if monitor is None or monitor.ensure_headroom(600):
                    loss_G_dists = losses["dists"](fake, target)
                else:
                    loss_G_dists = 0
            loss_G_nuclei = losses["nuclei"](fake, target) if "nuclei" in losses and losses["nuclei"] is not None else 0
            loss_G_color = losses["color"](fake, target) if "color" in losses else 0
            loss_G_grad = losses["grad"](fake, target) if "grad" in losses else 0
            loss_G = (weights["gan"]*loss_G_gan + weights["l1"]*loss_G_l1 +
                      weights["ssim"]*loss_G_ssim + weights["dists"]*loss_G_dists +
                      weights["nuclei"]*loss_G_nuclei + weights["color"]*loss_G_color +
                      weights["grad"]*loss_G_grad)
            loss_G = loss_G / grad_accum
            loss_G.backward()
            if (step+1) % grad_accum == 0:
                optimG.step()
                optimG.zero_grad()
            g_loss_accum += loss_G.item() * grad_accum
            d_loss_accum += d_loss_snapshot
            # component snapshots (raw, unweighted — multiply by weights at print time if needed)
            try:
                gan_accum += float(loss_G_gan.item())
            except Exception:
                pass
            try:
                l1_accum += float(loss_G_l1.item() if not isinstance(loss_G_l1, (int, float)) else loss_G_l1)
            except Exception:
                pass
            try:
                ssim_accum += float(loss_G_ssim.item() if hasattr(loss_G_ssim, "item") else loss_G_ssim)
            except Exception:
                pass
            try:
                nuclei_accum += float(loss_G_nuclei.item() if hasattr(loss_G_nuclei, "item") else loss_G_nuclei)
            except Exception:
                pass
            try:
                color_accum += float(loss_G_color.item() if hasattr(loss_G_color, "item") else loss_G_color)
            except Exception:
                pass
            try:
                grad_accum_loss += float(loss_G_grad.item() if hasattr(loss_G_grad, "item") else loss_G_grad)
            except Exception:
                pass
            seen += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM] G step OOM at batch {step}: {e} -> skip + gc")
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                optimG.zero_grad()
                continue
            else: raise
        # periodic monitor + gc
        if monitor and (step+1) % 20 == 0:
            monitor.check(f"train:step{step+1}")
            import gc; gc.collect()
        # free per step
        del he, dapi, src, src01, target, fake
    # flush remaining grads
    if seen % grad_accum != 0:
        optimG.step(); optimG.zero_grad()
        optimD.step(); optimD.zero_grad()
    denom = max(1, seen)
    return {
        "g": g_loss_accum / denom,
        "d": d_loss_accum / denom,
        "gan": gan_accum / denom,
        "l1": l1_accum / denom,
        "ssim": ssim_accum / denom,
        "nuclei": nuclei_accum / denom,
        "color": color_accum / denom,
        "grad": grad_accum_loss / denom,
    }


def validate(G, loader, device, direction):
    """Fixed-subset validation. Lower score is better (L1 + 0.25*(1-SSIM)).

    SSIM is measured with skimage (same definition as eval_offline), NOT with
    ``SSIMLoss``. ``SSIMLoss`` returns 1-SSIM (or a cosine proxy when kornia is
    absent), so using it here inverted the best-checkpoint selection.
    """
    from skimage.metrics import structural_similarity

    G.eval()
    l1_fn = nn.L1Loss()
    l1_sum = ssim_sum = psnr_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            he = batch["he"].to(device)
            dapi = batch["dapi"].to(device)
            src, _, target = prepare_pair({"he": he, "dapi": dapi}, direction)
            fake = G(src)
            l1 = l1_fn(fake, target)
            mse = torch.mean((fake - target) ** 2).item()
            fake_np = fake[0].detach().cpu().numpy()
            real_np = target[0].detach().cpu().numpy()
            if fake_np.shape[0] == 1:
                ssim = float(structural_similarity(real_np[0], fake_np[0], data_range=1.0))
            else:
                ssim = float(structural_similarity(real_np, fake_np, data_range=1.0, channel_axis=0))
            l1_sum += l1.item()
            ssim_sum += ssim
            psnr_sum += 10.0 * math.log10(1.0 / max(mse, 1e-12))
            n += 1
    G.train()
    n = max(1, n)
    l1 = l1_sum / n
    ssim = ssim_sum / n
    return {
        "l1": l1,
        "ssim": ssim,
        "psnr": psnr_sum / n,
        "score": l1 + 0.25 * (1.0 - ssim),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/hemit_dual_branch.yaml")
    ap.add_argument("--dummy", action="store_true", help="quick smoke test (2 batches, no checkpoint)")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume (auto-detects latest if --resume without value)")
    ap.add_argument("--resume-epoch", type=int, default=None, help="override start epoch")
    ap.add_argument("--optimize-for", type=str, default="lowram", choices=["lowram","full"], help="lowram throttles batch/DISTS for 3.7GB box")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch size")
    ap.add_argument("--grad-accum", type=int, default=None, help="override grad accum")
    ap.add_argument("--h5", type=str, default=None, help="override training HDF5 path")
    ap.add_argument("--checkpoint-dir", type=str, default=None, help="override checkpoint directory")
    ap.add_argument("--epochs", type=int, default=None, help="override number of epochs")
    ap.add_argument("--max-steps", type=int, default=None, help="stop each epoch after N steps (smoke/timing)")
    ap.add_argument("--direction", type=str, default=None, choices=["he2dapi","dapi2he"], help="translation direction")
    ap.add_argument("--val-h5", type=str, default=None, help="validation HDF5 (enables best-checkpoint selection)")
    ap.add_argument("--val-num", type=int, default=None, help="number of fixed validation crops")
    ap.add_argument("--val-every", type=int, default=None, help="validate every N epochs")
    args = ap.parse_args()

    # Resource monitor early
    try:
        from src.utils.resource_monitor import ResourceMonitor
        mon = ResourceMonitor(budget_gb=1.5, log_path="wiki/06-training/profile_paired.json")
    except Exception:
        mon = None
    if mon: mon.check("train_paired:start")

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    print(f"[CONFIG] {cfg.get('model',{}).get('name','pix2pix')} | data {cfg.get('data',{})}")

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = model_cfg.get("loss", {})
    if args.h5:
        data_cfg["processed"] = args.h5
    if args.checkpoint_dir:
        train_cfg["checkpoint_dir"] = args.checkpoint_dir
    direction = (args.direction or data_cfg.get("direction", "he2dapi")).lower()
    if direction not in {"he2dapi", "dapi2he"}:
        raise SystemExit(f"unknown direction: {direction}")
    cfg.setdefault("data", {})["direction"] = direction
    print(f"[DIRECTION] {direction}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device} (optimize_for={args.optimize_for})")
    if mon: print(f"[RAM] avail {mon.check('train:device', verbose=False)['ram']['avail_mb']:.0f}MB, advised batch {mon.advise_batch(train_cfg.get('batch_size',2))}")

    # Checkpoint dir early for resume detection
    ckpt_dir = Path(train_cfg.get("checkpoint_dir","resources/models/checkpoints/he2dapi"))
    if args.dummy:
        # never touch real checkpoints during a smoke test (would clobber live training)
        import tempfile
        ckpt_dir = Path(tempfile.mkdtemp(prefix="he_dapi_dummy_paired_"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Datasets - low-RAM: support subset via data_cfg lowram_subset or dummy
    h5_path = data_cfg.get("processed", "data/processed/hemit_512.h5")
    root_dir = data_cfg.get("root", "data/raw/hemit")
    # if requested H5 missing, fallback to available split
    if not Path(h5_path).exists():
        for fallback in ["data/processed/hemit_train_512.h5","data/processed/maxpmx_train_512.h5","data/processed/hemit_512.h5"]:
            if Path(fallback).exists():
                print(f"[FALLBACK] {h5_path} not found -> using {fallback}")
                h5_path = fallback
                break
    indices = None
    if args.dummy:
        indices = range(4)
    elif data_cfg.get("lowram_subset"):
        indices = range(int(data_cfg["lowram_subset"]))
        print(f"[LOWRAM] Using subset {len(indices)} from {h5_path}")
    elif args.optimize_for=="lowram" and Path(h5_path).exists():
        # if avail < 1GB, auto subset 500 to keep epoch quick + low RAM
        if mon and not mon.ensure_headroom(800):
            indices = range(500)
            print(f"[LOWRAM] Low avail, auto subset 500")

    ds = PairedHEMITDataset(h5_path=h5_path, root_dir=root_dir,
                            patch_size=data_cfg.get("patch_size",512),
                            stain_norm=data_cfg.get("stain_norm","none"),
                            augment=data_cfg.get("augment", True),
                            augmentation=data_cfg.get("augmentation"),
                            indices=indices,
                            max_cache_mb=train_cfg.get("max_cache_mb",4))
    loader_batch = args.batch_size or train_cfg.get("batch_size",2)
    if args.dummy:
        loader_batch = 1  # keep the smoke test inside the 3GB address-space cap
    # oom guard: advise down
    if mon:
        advised = mon.advise_batch(loader_batch)
        if advised < loader_batch:
            print(f"[OOM GUARD] batch {loader_batch} -> {advised} (RAM pressure)")
            loader_batch = advised
    loader = DataLoader(ds, batch_size=loader_batch, shuffle=True, num_workers=train_cfg.get("num_workers",0))

    # Fixed validation subset for best-checkpoint selection (no augmentation).
    val_h5 = args.val_h5 or train_cfg.get("val_h5")
    val_num = args.val_num if args.val_num is not None else int(train_cfg.get("val_num", 0))
    val_every = args.val_every if args.val_every is not None else int(train_cfg.get("val_every", 1))
    val_loader = None
    if val_h5 and val_num > 0:
        if not Path(val_h5).exists():
            print(f"[VAL WARN] {val_h5} not found -> best-checkpoint selection disabled")
        else:
            val_ds = PairedHEMITDataset(h5_path=val_h5, root_dir=root_dir,
                                        patch_size=data_cfg.get("patch_size",512),
                                        stain_norm=data_cfg.get("stain_norm","none"),
                                        augment=False,
                                        indices=range(val_num),
                                        max_cache_mb=min(int(train_cfg.get("max_cache_mb",4)), 16))
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
            print(f"[VAL] {val_num} fixed crops from {val_h5} every {val_every} epoch(s)")

    # Models: PatchGAN is conditioned on source (1 or 3 ch) + target (1 or 3 ch).
    in_ch, out_ch = (3, 1) if direction == "he2dapi" else (1, 3)
    G = UNetGenerator(in_ch=in_ch, out_ch=out_ch, ngf=model_cfg.get("generator",{}).get("ngf",64)).to(device)
    D = PatchGANDiscriminator(in_ch=4).to(device)
    optimG = torch.optim.Adam(G.parameters(), lr=train_cfg.get("lr",2e-4), betas=(0.5,0.999))
    optimD = torch.optim.Adam(D.parameters(), lr=train_cfg.get("lr",2e-4), betas=(0.5,0.999))
    # Losses - DISTS only when weighted; GPU is always allowed, CPU needs opt-in.
    use_dists = loss_cfg.get("dists_weight", 0.0) > 0 and (
        torch.cuda.is_available() or (train_cfg.get("dists_on_cpu", False) and args.optimize_for != "lowram")
    )
    if not use_dists:
        print("[LOSS] DISTS disabled (weight 0 or CPU without opt-in) -> L1+SSIM+optional color/grad")
    use_nuclei = loss_cfg.get("nuclei_aux_weight", 2.0) > 0
    use_color = loss_cfg.get("color_weight", 0.0) > 0
    use_grad = loss_cfg.get("grad_weight", 0.0) > 0
    losses = {
        "ssim": SSIMLoss(),
        "dists": DISTSLoss() if use_dists else None,
        "nuclei": NucleiDiceLoss(
            thresh=float(loss_cfg.get("nuclei_thresh", 0.1)),
            temperature=float(loss_cfg.get("nuclei_temp", 0.05)),
        ) if use_nuclei else None,
        "color": ColorStatsLoss() if use_color else None,
        "grad": GradientLoss() if use_grad else None,
    }
    # filter None
    losses = {k:v for k,v in losses.items() if v is not None}
    weights = {
        "gan": loss_cfg.get("gan_weight",1.0),
        "l1": loss_cfg.get("l1_weight",100.0),
        "ssim": loss_cfg.get("ssim_weight",10.0),
        "dists": loss_cfg.get("dists_weight",5.0),
        "nuclei": loss_cfg.get("nuclei_aux_weight",2.0),
        "color": loss_cfg.get("color_weight",0.0),
        "grad": loss_cfg.get("grad_weight",0.0),
    }

    # --- Resume logic (checkpoint every epoch, resumable) ---
    start_epoch = 0
    # auto-resume: find latest checkpoint if --resume flag given without path or if ckpt exists
    resume_path = None
    if args.resume is not None:
        # if args.resume is empty string or "auto", pick latest
        if args.resume in ("", "auto"):
            candidates = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"), key=lambda p: p.stat().st_mtime)
            if candidates:
                resume_path = candidates[-1]
            elif (ckpt_dir / "checkpoint_last.pt").exists():
                resume_path = ckpt_dir / "checkpoint_last.pt"
        else:
            resume_path = Path(args.resume)
    else:
        # auto-detect last checkpoint even without --resume (crash recovery)
        if (ckpt_dir / "checkpoint_last.pt").exists():
            resume_path = ckpt_dir / "checkpoint_last.pt"
    if resume_path and Path(resume_path).exists():
        ckpt = None
        try:
            ckpt = torch.load(str(resume_path), map_location=device)
        except Exception as e:
            print(f"[RESUME WARN] Could not read {resume_path}: {e}, starting from 0")
        if ckpt is not None:
            ckpt_direction = ckpt.get("config", {}).get("data", {}).get("direction")
            # Legacy checkpoints predate the direction field (None). Only
            # enforce a mismatch when the checkpoint records a direction.
            if ckpt_direction and ckpt_direction != direction:
                raise SystemExit(f"checkpoint direction {ckpt_direction} != requested {direction}")
            try:
                G.load_state_dict(ckpt.get("G", ckpt.get("generator", ckpt)))
                if "D" in ckpt:
                    D.load_state_dict(ckpt["D"])
                if "optimG" in ckpt:
                    optimG.load_state_dict(ckpt["optimG"])
                if "optimD" in ckpt:
                    optimD.load_state_dict(ckpt["optimD"])
                for pg in optimG.param_groups:
                    pg["lr"] = train_cfg.get("lr", 2e-4)
                for pg in optimD.param_groups:
                    pg["lr"] = train_cfg.get("lr", 2e-4)
                start_epoch = ckpt.get("epoch", 0) + 1
                if args.resume_epoch is not None:
                    start_epoch = args.resume_epoch
                print(f"[RESUME] Loaded {resume_path} -> start_epoch {start_epoch}")
            except Exception as e:
                print(f"[RESUME WARN] Failed to load {resume_path}: {e}, starting from 0")

    epochs = 2 if args.dummy else (args.epochs or train_cfg.get("epochs",200))
    save_every = train_cfg.get("save_every", 1)  # checkpoint every N epochs
    grad_accum = args.grad_accum or train_cfg.get("grad_accum", 1)
    oom_guard = train_cfg.get("oom_guard", True)
    print(f"[TRAIN] epochs {epochs} batch {loader_batch} grad_accum {grad_accum} effective {loader_batch*grad_accum} | oom_guard {oom_guard} | device {device}")
    best_score = float("inf")
    best_path = None
    existing_best = ckpt_dir / "checkpoint_best.pt"
    if existing_best.exists():
        try:
            best_ckpt = torch.load(str(existing_best), map_location="cpu")
            if best_ckpt.get("val_score") is not None:
                best_score = float(best_ckpt["val_score"])
                best_path = existing_best
                print(f"[BEST] Existing {existing_best} score {best_score:.4f}")
        except Exception as e:
            print(f"[BEST WARN] Could not read {existing_best}: {e}")
    for epoch in range(start_epoch, epochs):
        if mon: mon.check(f"train:epoch{epoch+1}:start")
        loss = train_one_epoch(G,D,loader,optimG,optimD,device,losses,weights, grad_accum=grad_accum, oom_guard=oom_guard, monitor=mon, max_steps=args.max_steps, direction=direction)
        # new dict form (g/d/components) — keep ckpt["loss"] as G for history continuity
        if isinstance(loss, dict):
            g_loss = loss["g"]
            print(f"[Epoch {epoch+1}/{epochs}] G loss {g_loss:.4f} D loss {loss['d']:.4f} (gan {loss['gan']:.4f} l1 {loss['l1']:.4f} ssim {loss['ssim']:.4f} nuclei {loss['nuclei']:.4f} color {loss.get('color',0):.4f} grad {loss.get('grad',0):.4f})")
        else:
            g_loss = float(loss)
            print(f"[Epoch {epoch+1}/{epochs}] G loss {g_loss:.4f}")
        if mon: mon.check(f"train:epoch{epoch+1}:end")

        val_metrics = None
        if val_loader is not None and ((epoch+1) % val_every == 0 or (epoch+1) == epochs):
            val_metrics = validate(G, val_loader, device, direction)
            print(f"[VAL] Epoch {epoch+1} L1 {val_metrics['l1']:.4f} SSIM {val_metrics['ssim']:.4f} "
                  f"PSNR {val_metrics['psnr']:.2f} score {val_metrics['score']:.4f} n {val_metrics['n']}")

        ckpt = {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optimG": optimG.state_dict(),
            "optimD": optimD.state_dict(),
            "loss": g_loss,
            "loss_detail": loss if isinstance(loss, dict) else {"g": g_loss},
            "val": val_metrics,
            "val_score": val_metrics["score"] if val_metrics else None,
            "config": cfg,
        }
        # Save resumable checkpoint every save_every + always keep last
        if (epoch+1) % save_every == 0 or (epoch+1) == epochs:
            # atomic save: write to tmp then rename
            tmp_path = ckpt_dir / f".tmp_checkpoint_epoch{epoch+1}.pt"
            final_path = ckpt_dir / f"checkpoint_epoch{epoch+1}.pt"
            last_path = ckpt_dir / "checkpoint_last.pt"
            torch.save(ckpt, tmp_path)
            tmp_path.rename(final_path)
            torch.save(ckpt, last_path)
            print(f"[CHECKPOINT] Epoch {epoch+1} -> {final_path} + {last_path} (resumable: --resume {last_path})")

        if val_metrics is not None and val_metrics["score"] < best_score:
            best_score = val_metrics["score"]
            best_path = ckpt_dir / "checkpoint_best.pt"
            tmp_best = ckpt_dir / ".tmp_checkpoint_best.pt"
            torch.save(ckpt, tmp_best)
            tmp_best.rename(best_path)
            torch.save(G.state_dict(), ckpt_dir / "generator_best.pt")
            print(f"[BEST] Epoch {epoch+1} score {best_score:.4f} -> {best_path} + generator_best.pt")

    # Save final generator only (for inference) + keep last
    ckpt_path = ckpt_dir / "generator_last.pt"
    torch.save(G.state_dict(), ckpt_path)
    print(f"[SAVE] Generator -> {ckpt_path}")
    if best_path:
        print(f"[SAVE] Best checkpoint -> {best_path} (score {best_score:.4f})")

    if args.export_onnx or train_cfg.get("export_onnx") or args.dummy:
        dummy = torch.randn(1,in_ch,512,512).to(device)
        in_name, out_name = ("he", "dapi") if direction == "he2dapi" else ("dapi", "he")
        onnx_path = ckpt_dir / f"{direction}_pix2pix.onnx"
        try:
            torch.onnx.export(G, dummy, str(onnx_path), input_names=[in_name], output_names=[out_name], dynamic_axes={in_name:{0:"batch"},out_name:{0:"batch"}})
            print(f"[ONNX] Exported -> {onnx_path}")
        except Exception as e:
            print(f"[ONNX] Export failed: {e}")

    # completion flag: tells cron/keepalive not to relaunch a finished run
    if not args.dummy:
        flag = Path("logs/train_done.flag")
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(f"finished {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} epochs={epochs}\n")
        print(f"[DONE] Training complete -> {flag}")

if __name__=="__main__":
    main()
