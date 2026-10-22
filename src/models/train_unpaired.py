"""
train_unpaired.py - CycleGAN / CUT for DAPI -> H&E (unpaired)
CycleGAN: two generators G: DAPI->HE, F: HE->DAPI + two discriminators Dx, Dy + cycle + identity
CUT variant: single G + NCE loss (simpler, less hallucination)

Run:
  python src/models/train_unpaired.py --config configs/cyclegan_dapi2he.yaml --dummy
"""
import argparse, time, yaml
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from .datasets import UnpairedDataset

class ResnetGenerator(nn.Module):
    """ResNet generator for CycleGAN: 1ch DAPI -> 3ch HE (or vice versa). Simplified 6 blocks."""
    def __init__(self, in_ch=1, out_ch=3, ngf=64, n_blocks=6):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, ngf, 7,1,3), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3,2,1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3,2,1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            layers += [ResBlock(ngf*4)]
        layers += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3,2,1,1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3,2,1,1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, out_ch, 7,1,3), nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return (self.net(x)+1)/2  # -> [0,1] for HE, careful for DAPI side handle outside

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,1,1), nn.InstanceNorm2d(ch)
        )
    def forward(self,x): return x + self.block(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,64,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,2,1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256,1,4,1,1)
        )
    def forward(self,x): return self.net(x)

def train_dummy(monitor=None, batch_size=2, grad_accum=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CycleGAN DUMMY] device {device} batch {batch_size} grad_accum {grad_accum}")
    if monitor: monitor.check("cyclegan:dummy:start")
    # Use H5 fallback if PNG missing (optimized path)
    ds = UnpairedDataset(he_root="data/raw/mhist", dapi_root="data/processed/hemit_train_512.h5", patch_size=256)
    from torch.utils.data import Subset
    ds = Subset(ds, range(min(4,len(ds))))
    # OOM guard batch
    if monitor:
        advised = monitor.advise_batch(batch_size)
        if advised < batch_size:
            print(f"[OOM GUARD] dummy batch {batch_size}->{advised}")
            batch_size = advised
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    G = ResnetGenerator(1,3).to(device)  # DAPI->HE
    F = ResnetGenerator(3,1).to(device)  # HE->DAPI
    Dx = PatchDiscriminator(3).to(device)
    Dy = PatchDiscriminator(1).to(device)
    optG = torch.optim.Adam(list(G.parameters())+list(F.parameters()), lr=2e-4, betas=(0.5,0.999))
    optD = torch.optim.Adam(list(Dx.parameters())+list(Dy.parameters()), lr=2e-4, betas=(0.5,0.999))
    for epoch in range(1):
        for step, batch in enumerate(loader):
            try:
                he = batch["he"].to(device)   # [-1,1] but we map? For dummy use [0,1]
                dapi = batch["dapi"].to(device) # [0,1]
                he01 = (he+1)/2
                fake_he = G(dapi)  # [0,1]
                fake_he_m11 = fake_he*2-1
                rec_dapi = F(fake_he_m11)
                # For smoke test, just run G
                loss_g = nn.L1Loss()(fake_he, he01) / grad_accum  # identity-ish
                loss_g.backward()
                if (step+1) % grad_accum == 0:
                    optG.step(); optG.zero_grad()
                print(f"  batch loss {loss_g.item()*grad_accum:.4f}")
                del he, dapi, fake_he, he01
                if monitor and step%5==0:
                    monitor.check(f"cyclegan:dummy:step{step}")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] dummy OOM: {e} -> gc")
                    import gc; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue
                raise
        print(f"[Epoch {epoch+1}] dummy step done")
    # Save
    Path("resources/models/checkpoints/dapi2he").mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), "resources/models/checkpoints/dapi2he/cyclegan_G_last.pt")
    print("[SAVE] G -> resources/models/checkpoints/dapi2he/cyclegan_G_last.pt")
    if monitor: monitor.check("cyclegan:dummy:end")


def _cyclegan_step(G, F, D_he, D_dapi, optimG, optimD, he01, dapi01,
                   weights, grad_accum, step, device):
    """One CycleGAN step. Domains are [0,1]; generators return [0,1]."""
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    fake_he = G(dapi01)
    fake_dapi = F(he01)
    rec_dapi = F(fake_he)
    rec_he = G(fake_dapi)
    # Identity loss only makes sense when both domains have the same channel
    # count. DAPI(1) -> H&E(3) is asymmetric, so it is disabled (weight 0).
    if he01.shape[1] == dapi01.shape[1]:
        id_he = G(he01)
        id_dapi = F(dapi01)
        loss_id = l1(id_he, he01) + l1(id_dapi, dapi01)
    else:
        loss_id = torch.zeros((), device=device)

    # --- Discriminators (detach generator outputs) ---
    loss_D_he = 0.5 * (mse(D_he(he01), torch.ones_like(D_he(he01))) +
                       mse(D_he(fake_he.detach()), torch.zeros_like(D_he(fake_he.detach()))))
    loss_D_dapi = 0.5 * (mse(D_dapi(dapi01), torch.ones_like(D_dapi(dapi01))) +
                         mse(D_dapi(fake_dapi.detach()), torch.zeros_like(D_dapi(fake_dapi.detach()))))
    loss_D = (loss_D_he + loss_D_dapi) / grad_accum
    loss_D.backward()

    # --- Generators ---
    loss_G_gan = (mse(D_he(fake_he), torch.ones_like(D_he(fake_he))) +
                  mse(D_dapi(fake_dapi), torch.ones_like(D_dapi(fake_dapi))))
    loss_cycle = l1(rec_dapi, dapi01) + l1(rec_he, he01)
    loss_G = (weights["gan"] * loss_G_gan +
              weights["cycle"] * loss_cycle +
              weights["identity"] * loss_id) / grad_accum
    loss_G.backward()

    if (step + 1) % grad_accum == 0:
        optimD.step(); optimG.step()
        optimD.zero_grad(); optimG.zero_grad()

    return {
        "g": float(loss_G.item() * grad_accum),
        "d": float(loss_D.item() * grad_accum),
        "gan": float(loss_G_gan.item()),
        "cycle": float(loss_cycle.item()),
        "identity": float(loss_id.item()),
    }


def train_cyclegan(cfg, args, mon=None):
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("model", {}).get("loss", {})
    gen_cfg = cfg.get("model", {}).get("generator", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CycleGAN] device {device}")

    ds = UnpairedDataset(
        he_root=data_cfg.get("he_root", "data/raw/mhist"),
        dapi_root=data_cfg.get("dapi_root", "data/processed/hemit_train_512.h5"),
        patch_size=data_cfg.get("patch_size", 256),
        he_h5=data_cfg.get("he_h5_fallback"),
        dapi_h5=data_cfg.get("dapi_h5_fallback"),
    )
    if len(ds) == 0:
        raise SystemExit("unpaired dataset is empty")
    batch_size = args.batch_size or train_cfg.get("batch_size", 2)
    grad_accum = args.grad_accum or train_cfg.get("grad_accum", 1)
    if mon:
        advised = mon.advise_batch(batch_size)
        if advised < batch_size:
            print(f"[OOM GUARD] batch {batch_size}->{advised}")
            batch_size = advised
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=train_cfg.get("num_workers", 0), drop_last=True)

    ngf = gen_cfg.get("ngf", 64)
    n_blocks = gen_cfg.get("n_blocks", 9)
    G = ResnetGenerator(1, 3, ngf=ngf, n_blocks=n_blocks).to(device)  # DAPI -> HE
    F = ResnetGenerator(3, 1, ngf=ngf, n_blocks=n_blocks).to(device)  # HE -> DAPI
    D_he = PatchDiscriminator(3).to(device)
    D_dapi = PatchDiscriminator(1).to(device)
    optimG = torch.optim.Adam(list(G.parameters()) + list(F.parameters()),
                              lr=train_cfg.get("lr", 2e-4), betas=(0.5, 0.999))
    optimD = torch.optim.Adam(list(D_he.parameters()) + list(D_dapi.parameters()),
                              lr=train_cfg.get("lr", 2e-4), betas=(0.5, 0.999))
    weights = {
        "gan": loss_cfg.get("gan_weight", 1.0),
        "cycle": loss_cfg.get("cycle_weight", 10.0),
        "identity": loss_cfg.get("identity_weight", 5.0),
    }

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "resources/models/checkpoints/dapi2he"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    resume_path = None
    if args.resume is not None:
        if args.resume in ("", "auto"):
            cands = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"), key=lambda p: p.stat().st_mtime)
            if cands:
                resume_path = cands[-1]
            elif (ckpt_dir / "checkpoint_last.pt").exists():
                resume_path = ckpt_dir / "checkpoint_last.pt"
        else:
            resume_path = Path(args.resume)
    elif (ckpt_dir / "checkpoint_last.pt").exists():
        resume_path = ckpt_dir / "checkpoint_last.pt"
    if resume_path and Path(resume_path).exists():
        try:
            ckpt = torch.load(str(resume_path), map_location=device)
            G.load_state_dict(ckpt["G"]); F.load_state_dict(ckpt["F"])
            if "D_he" in ckpt: D_he.load_state_dict(ckpt["D_he"])
            if "D_dapi" in ckpt: D_dapi.load_state_dict(ckpt["D_dapi"])
            if "optimG" in ckpt: optimG.load_state_dict(ckpt["optimG"])
            if "optimD" in ckpt: optimD.load_state_dict(ckpt["optimD"])
            start_epoch = ckpt.get("epoch", 0) + 1
            if getattr(args, "resume_epoch", None) is not None:
                start_epoch = args.resume_epoch
            print(f"[RESUME] {resume_path} -> start_epoch {start_epoch}")
        except Exception as exc:
            print(f"[RESUME WARN] {resume_path}: {exc} -> starting fresh")

    epochs = args.epochs or train_cfg.get("epochs", 200)
    save_every = train_cfg.get("save_every", 1)
    max_steps = getattr(args, "max_steps", None)
    log_every = train_cfg.get("log_every", 100)
    print(f"[TRAIN] epochs {epochs} batch {batch_size} grad_accum {grad_accum} "
          f"effective {batch_size * grad_accum} steps/epoch {len(loader)}")

    for epoch in range(start_epoch, epochs):
        G.train(); F.train(); D_he.train(); D_dapi.train()
        t0 = time.time()
        acc = {"g": 0.0, "d": 0.0, "gan": 0.0, "cycle": 0.0, "identity": 0.0}
        seen = 0
        for step, batch in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            try:
                he01 = (batch["he"].to(device) + 1) / 2  # [-1,1] -> [0,1]
                dapi01 = batch["dapi"].to(device)         # [0,1]
                stats = _cyclegan_step(G, F, D_he, D_dapi, optimG, optimD,
                                       he01, dapi01, weights, grad_accum, step, device)
                for k in acc:
                    acc[k] += stats[k]
                seen += 1
                if log_every and (step + 1) % log_every == 0:
                    print(f"  step {step + 1}/{len(loader)} G {stats['g']:.3f} "
                          f"D {stats['d']:.3f} cycle {stats['cycle']:.3f} "
                          f"id {stats['identity']:.3f}", flush=True)
                del he01, dapi01
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[OOM] step {step}: {exc} -> gc + skip")
                    optimG.zero_grad(); optimD.zero_grad()
                    import gc; gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise RuntimeError(f"step {step}: {exc}") from exc
        denom = max(seen, 1)
        print(f"[Epoch {epoch+1}/{epochs}] G {acc['g']/denom:.4f} D {acc['d']/denom:.4f} "
              f"(gan {acc['gan']/denom:.4f} cycle {acc['cycle']/denom:.4f} "
              f"id {acc['identity']/denom:.4f}) {time.time()-t0:.0f}s", flush=True)
        if mon:
            mon.check(f"cyclegan:epoch{epoch+1}:end")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt = {
                "epoch": epoch, "G": G.state_dict(), "F": F.state_dict(),
                "D_he": D_he.state_dict(), "D_dapi": D_dapi.state_dict(),
                "optimG": optimG.state_dict(), "optimD": optimD.state_dict(),
                "config": cfg,
            }
            tmp = ckpt_dir / f".tmp_checkpoint_epoch{epoch+1}.pt"
            torch.save(ckpt, tmp)
            tmp.rename(ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")
            torch.save(ckpt, ckpt_dir / "checkpoint_last.pt")
            print(f"[CHECKPOINT] Epoch {epoch+1} -> {ckpt_dir / 'checkpoint_last.pt'}", flush=True)

    for name, state in (("cyclegan_G_last.pt", G.state_dict()),
                        ("generator_last.pt", G.state_dict())):
        torch.save(state, ckpt_dir / name)
        print(f"[SAVE] {name} -> {ckpt_dir / name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/cyclegan_dapi2he.yaml")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume (auto-detects latest)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop each epoch after N steps")
    ap.add_argument("--resume-epoch", type=int, default=None)
    ap.add_argument("--checkpoint-dir", type=str, default=None, help="override checkpoint directory")
    ap.add_argument("--optimize-for", type=str, default="lowram", choices=["lowram","full"])
    args = ap.parse_args()
    # monitor
    try:
        from src.utils.resource_monitor import ResourceMonitor
        mon = ResourceMonitor(budget_gb=1.5, log_path="wiki/06-training/profile_unpaired.json")
    except: mon=None
    if mon: mon.check("cyclegan:start")
    if args.dummy:
        cfg_tmp = {}
        if Path(args.config).exists():
            with open(args.config) as f: cfg_tmp=yaml.safe_load(f)
        train_cfg_tmp = cfg_tmp.get("training",{}) if cfg_tmp else {}
        bs = args.batch_size or train_cfg_tmp.get("batch_size",2)
        ga = args.grad_accum or train_cfg_tmp.get("grad_accum",4)
        train_dummy(monitor=mon, batch_size=bs, grad_accum=ga)
        return
    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f: cfg = yaml.safe_load(f)
    if args.checkpoint_dir:
        cfg.setdefault("training", {})["checkpoint_dir"] = args.checkpoint_dir
    print(f"[CONFIG] model={cfg.get('model',{}).get('name')} data={cfg.get('data',{})}")
    train_cyclegan(cfg, args, mon=mon)

if __name__=="__main__":
    main()
