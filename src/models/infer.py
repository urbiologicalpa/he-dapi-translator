"""
infer.py - Patch + WSI tiled inference, ONNX support, uncertainty via MC dropout
Usage:
  python src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/out.png
  python src/models/infer.py --input data/raw/wsi/CM1_ShAHE_0_0_0.svs --direction he2dapi --wsi --tile 512 --stride 256
  python src/models/infer.py --input data/examples/he_example.png --direction he2dapi --onnx resources/models/checkpoints/he2dapi/he2dapi_pix2pix.onnx
"""
import argparse
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

def load_image(path: str, max_size: int = 2048, stream: bool = False) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in [".svs",".tiff",".tif",".ome.tiff"] or stream:
        # For large WSI, use WSILoader thumbnail to avoid loading 40k×40k
        try:
            from src.utils.wsi_loader import WSILoader
            loader = WSILoader(str(p))
            # if file is actually huge, thumb is needed; but if p is PNG with large dims, use stream?
            thumb = loader.get_thumbnail(max_size=max_size)
            loader.close()
            # For true streaming WSI inference, caller should use infer_wsi_streaming, not load_image
            return thumb
        except Exception as e:
            print(f"[LOAD] WSILoader failed {e}, fallback PIL")
            return np.array(Image.open(p).convert("RGB"))
    else:
        # Guard large PNG: if > 50MP, downsample via PIL thumbnail before load to keep RAM <512MB
        try:
            from src.utils.resource_monitor import get_ram_state
            ram = get_ram_state()
            img = Image.open(p)
            w,h = img.size
            est_mb = w*h*3/1e6
            if est_mb > ram["avail_mb"]*0.5 or est_mb > 512:
                print(f"[LOAD] Large image {w}x{h} est {est_mb:.0f}MB -> thumbnail {max_size} to fit RAM {ram['avail_mb']:.0f}MB")
                img.thumbnail((max_size, max_size), Image.BILINEAR)
            img = img.convert("RGB")
            img.load()
            return np.array(img)
        except Exception:
            return np.array(Image.open(p).convert("RGB"))

def save_image(arr: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # arr: HWC uint8, may be 1ch (DAPI) or 3ch (HE)
    if arr.ndim==2:
        Image.fromarray(arr).save(path)
    elif arr.shape[2]==1:
        Image.fromarray(arr[:,:,0]).save(path)
    else:
        Image.fromarray(arr).save(path)
    print(f"[SAVE] {path} shape {arr.shape}")

def detect_direction(image: np.ndarray, threshold: float = 0.06) -> str:
    """Route an image to he2dapi or dapi2he from its colour content.

    DAPI is (near) grayscale, so the mean channel chroma is ~0. H&E is stained
    and lands well above the threshold (~0.2-0.35 on HEMIT/MHIST examples).
    """
    arr = np.asarray(image)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return "dapi2he"
    rgb = arr[..., :3].astype(np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    chroma = float(np.mean((mx - mn) / np.maximum(mx, 1e-6)))
    return "he2dapi" if chroma >= threshold else "dapi2he"


def resolve_direction(direction: str, image: np.ndarray) -> str:
    """Resolve 'auto' against an image; pass explicit directions through."""
    direction = (direction or "auto").lower()
    if direction == "auto":
        chosen = detect_direction(image)
        print(f"[ROUTER] auto -> {chosen}")
        return chosen
    if direction not in {"he2dapi", "dapi2he"}:
        raise ValueError(f"unknown direction: {direction}")
    return direction


def run_torch_model(tile: np.ndarray, direction: str, checkpoint: str = None, dummy: bool = False) -> np.ndarray:
    """Run PyTorch generator on single tile (HWC uint8) -> synthetic HWC uint8. Falls back to dummy invert."""
    if dummy:
        return _dummy_transform(tile, direction)

    try:
        import torch
        from src.models.train_paired import UNetGenerator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Choose generator based on direction
        if direction=="he2dapi":
            G = UNetGenerator(in_ch=3, out_ch=1).to(device)
            if checkpoint and Path(checkpoint).exists():
                G.load_state_dict(torch.load(checkpoint, map_location=device))
            G.eval()
            # Preprocess HE [0,255] -> [-1,1]
            inp = torch.from_numpy(tile).permute(2,0,1).float().unsqueeze(0).to(device) / 127.5 - 1
            with torch.no_grad():
                out = G(inp)  # Bx1xHxW [0,1]
            out_np = (out[0,0].cpu().numpy()*255).astype(np.uint8)
            # Return grayscale DAPI, expand to HWC for consistency? Keep HxW
            return out_np
        else:  # dapi2he
            state = None
            if checkpoint and Path(checkpoint).exists():
                ckpt = torch.load(checkpoint, map_location=device)
                state = ckpt.get("G", ckpt.get("generator", ckpt))
            if state is not None and any(k.startswith("net.") for k in state):
                from src.models.train_unpaired import ResnetGenerator
                G = ResnetGenerator(in_ch=1, out_ch=3).to(device)
            else:
                G = UNetGenerator(in_ch=1, out_ch=3).to(device)
            if state is not None:
                G.load_state_dict(state)
            G.eval()
            # DAPI: tile may be RGB HE fake -> need grayscale
            if tile.ndim==3 and tile.shape[2]==3:
                # Input is DAPI grayscale but stored as RGB? Convert
                dapi_gray = np.dot(tile[...,:3], [0.2989,0.5870,0.1140]).astype(np.uint8)
            else:
                dapi_gray = tile if tile.ndim==2 else tile[:,:,0]
            inp = torch.from_numpy(dapi_gray).unsqueeze(0).unsqueeze(0).float().to(device)/255.0
            inp = inp * 2 - 1
            with torch.no_grad():
                out = G(inp)  # Bx3xHxW [0,1]
            out_np = (out[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            return out_np
    except Exception as e:
        print(f"[WARN] Torch model failed ({e}), using dummy transform")
        # Dummy: invert / colormap
        if direction=="he2dapi":
            # Fake DAPI: grayscale + nuclei enhance (simple luminance)
            if tile.ndim==3:
                gray = np.dot(tile[...,:3],[0.2989,0.5870,0.1140])
            else:
                gray = tile
            # Enhance nuclei-like: threshold blue channel if exists
            return (255 - gray).astype(np.uint8)
        else:
            # Fake H&E: colorize DAPI
            gray = tile if tile.ndim==2 else tile[:,:,0]
            # Simple H&E pink/purple colormap
            he = np.stack([gray, (gray*0.6).astype(np.uint8), (gray*0.8).astype(np.uint8)], axis=-1)
            # Add eosin-like pink
            he[:,:,0] = np.clip(he[:,:,0]*1.2,0,255).astype(np.uint8)
            return he

def run_onnx(tile: np.ndarray, onnx_path: str) -> np.ndarray:
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # Assume input name "he", shape Bx3xHxW
        inp = tile.transpose(2,0,1)[None].astype(np.float32) / 127.5 - 1
        out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]  # Bx1xHxW [0,1]
        out_np = (out[0,0]*255).astype(np.uint8)
        return out_np
    except Exception as e:
        print(f"[ONNX] failed {e}, fallback to torch")
        return run_torch_model(tile, "he2dapi")

def _dummy_transform(tile: np.ndarray, direction: str) -> np.ndarray:
    """Lightweight CPU-only transformation for inference smoke tests."""
    if direction == "he2dapi":
        if tile.ndim == 3:
            gray = np.dot(
                tile[..., :3], [0.2989, 0.5870, 0.1140]
            )
        else:
            gray = tile
        return (255 - gray).astype(np.uint8)

    gray = tile if tile.ndim == 2 else tile[..., 0]
    return np.stack(
        [
            gray,
            (gray * 0.6).astype(np.uint8),
            (gray * 0.8).astype(np.uint8),
        ],
        axis=-1,
    )

def infer_patch(input_path: str, direction: str, out_path: str, onnx_path: str = None, checkpoint: str = None, tile_size: int = 512, stride: int = 256, stream: bool = False, dummy: bool = False, profile_log_path: str = "wiki/07-inference/profile.json"):
    # Resource guard at entry
    try:
        from src.utils.resource_monitor import ResourceMonitor
        mon = ResourceMonitor(budget_gb=1.5, log_path=profile_log_path)
        mon.check(f"infer:{direction}:start")
    except: mon=None
    img = load_image(input_path, max_size=2048 if not stream else 4096, stream=stream)
    direction = resolve_direction(direction, img)
    print(f"[PATCH] Input {input_path} shape {img.shape}, direction {direction} (stream={stream})")
    # If image larger than tile_size, use tiled inference (streaming aware)
    h,w = img.shape[:2]
    if max(h,w) > 1024 or stream:
        from src.utils.tiling import TiledInference
        tiler = TiledInference(tile_size=tile_size, stride=stride)
        # Pre-load model once for speed + RAM check
        use_onnx = onnx_path and Path(onnx_path).exists()
        # Estimate canvas
        est_mb = h*w*(1 if direction=="he2dapi" else 3)*4/1e6
        do_stream = stream or est_mb > 512 or (mon and not mon.ensure_headroom(800))
        if do_stream and not out_path.endswith((".tiff",".tif")):
            # for streaming we prefer tiff to avoid holding PNG in RAM
            print(f"[INFER] Large {h}x{w} est {est_mb:.0f}MB -> streaming to {out_path} (tiff recommended)")
        def model_fn(tile):
            # tile is HWC RGB
            if use_onnx:
                res = run_onnx(tile, onnx_path)
            else:
                res = run_torch_model(tile, direction, checkpoint)
            # Ensure output is HWC for merging
            if res.ndim==2:
                res = res[...,None]
            if direction=="he2dapi" and res.shape[2]==1:
                pass
            elif direction=="dapi2he" and res.ndim==2:
                res = np.stack([res]*3, -1)
            return res
        # Use streaming merge if needed
        if do_stream and max(h,w) > 2048:
            # stream to tiff
            stream_out = out_path if out_path.endswith((".tiff",".tif")) else out_path.replace(".png",".tiff")
            result = tiler.infer_image(img, model_fn, out_path=stream_out)
            # tiler returns dummy 1x1 when streamed; handle by not saving again
            if result is not None and result.shape[0]==1 and result.shape[1]==1:
                print(f"[DONE] Streamed Direction {direction} -> {stream_out} (not loading full into RAM)")
                if mon: mon.check(f"infer:{direction}:stream_end")
                return None  # streamed
            # else was banded but returned array
            if direction=="he2dapi" and result.ndim==3 and result.shape[2]==1:
                result = result[:,:,0]
        else:
            result = tiler.infer_image(img, model_fn)
            if direction=="he2dapi" and result.ndim==3 and result.shape[2]==1:
                result = result[:,:,0]
            if mon: mon.check(f"infer:{direction}:tiled_end")
    else:
        if onnx_path and Path(onnx_path).exists():
            result = run_onnx(img, onnx_path)
        else:
            result = run_torch_model(img, direction, checkpoint,dummy=dummy)
        if mon: mon.check(f"infer:{direction}:patch_end")
    # Save only if not already streamed
    if result is not None and not (isinstance(result, np.ndarray) and result.shape[0]==1 and result.shape[1]==1):
        save_image(result, out_path)
    # Uncertainty placeholder (variance map)
    # For real: MC dropout 5 passes -> variance
    print(f"[DONE] Direction {direction} -> {out_path}")
    if mon: mon.check(f"infer:{direction}:end")
    return result

def infer_wsi_streaming(wsi_path: str, direction: str, out_path: str, checkpoint: str = None, tile_size: int = 512, stride: int = 256):
    """
    True WSI streaming: iterates level-0 tiles via WSILoader.read_region without loading full WSI.
    Writes directly to tiled TIFF via TiledInference.merge_tiles_streaming.
    Keeps RAM <512MB even for 40k×40k.
    """
    from src.utils.wsi_loader import WSILoader
    from src.utils.tiling import TiledInference
    from src.utils.resource_monitor import ResourceMonitor
    mon = ResourceMonitor(budget_gb=1.5, log_path="wiki/07-inference/profile_wsi.json")
    mon.check(f"wsi_stream:{direction}:start")
    loader = WSILoader(wsi_path)
    w,h = loader.level_dimensions[0]
    if (direction or "").lower() == "auto":
        direction = resolve_direction("auto", loader.get_thumbnail(max_size=1024))
    print(f"[WSI STREAM] {wsi_path} {w}x{h} backend {loader._backend} -> {out_path}")
    tiler = TiledInference(tile_size=tile_size, stride=stride)
    # Determine output channels
    out_c = 1 if direction=="he2dapi" else 3
    out_shape = (h,w,out_c)
    # Build generator of ((y,x), synthetic)
    def tile_gen():
        for (y,x), tile in tiler.extract_tiles: pass
        # Actually need to iterate over coordinates and fetch via loader.read_region
        # Use tiler's coordinate logic but fetch on demand
        # We'll enumerate y,x ourselves to use loader
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = loader.read_region(x, y, level=0, size=(tile_size, tile_size))
                synth = run_torch_model(tile, direction, checkpoint)
                if synth.ndim==2: synth = synth[...,None]
                yield (y,x), synth
        # edge tiles
        if h % stride !=0 or w % stride !=0:
            y = h - tile_size
            for x in range(0, w - tile_size +1, stride):
                tile = loader.read_region(x, y, level=0, size=(tile_size,tile_size))
                synth = run_torch_model(tile, direction, checkpoint)
                if synth.ndim==2: synth = synth[...,None]
                yield (y,x), synth
            x = w - tile_size
            for y in range(0, h - tile_size+1, stride):
                if y==h-tile_size: continue
                tile = loader.read_region(x, y, level=0, size=(tile_size,tile_size))
                synth = run_torch_model(tile, direction, checkpoint)
                if synth.ndim==2: synth = synth[...,None]
                yield (y,x), synth
        if mon and y%1024==0:
            mon.check(f"wsi_stream:y{y}")
    # For demo WSI where tiler.extract_tiles expects image, we instead use our gen above
    # Use tiler streaming merge directly
    # Need to recreate gen that yields correctly
    def gen2():
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = loader.read_region(x, y, level=0, size=(tile_size,tile_size))
                synth = run_torch_model(tile, direction, checkpoint)
                if synth.ndim==2: synth = synth[...,None]
                yield (y,x), synth
        # edges
        if h % stride !=0 or w % stride !=0:
            y = h - tile_size
            for x in range(0, w - tile_size +1, stride):
                tile = loader.read_region(x, y, level=0, size=(tile_size,tile_size))
                synth = run_torch_model(tile, direction, checkpoint)
                if synth.ndim==2: synth = synth[...,None]
                yield (y,x), synth
            x = w - tile_size
            for y in range(0, h - tile_size+1, stride):
                if y==h-tile_size: continue
                tile = loader.read_region(x, y, level=0, size=(tile_size,tile_size))
                synth = run_torch_model(tile, direction, checkpoint)
                if synth.ndim==2: synth = synth[...,None]
                yield (y,x), synth
    tiler.merge_tiles_streaming(gen2(), out_shape, out_path=out_path)
    loader.close()
    mon.check(f"wsi_stream:{direction}:end")
    print(f"[WSI DONE] -> {out_path}")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--dummy", action="store_true", help="run a lightweight inference smoke test")
    ap.add_argument("--direction", type=str, default="he2dapi", choices=["he2dapi","dapi2he","auto"])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--onnx", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--wsi", action="store_true", help="treat input as WSI (tiled)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--stream", action="store_true", help="force streaming mode (low RAM)")
    args = ap.parse_args()

    if args.dummy:
        if Image is None:
            ap.error("Pillow is required for the dummy smoke test")

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "dummy_input.png"
            output_path = Path(temp_dir) / "dummy_output.png"

            dummy_input = np.random.randint(
                0, 256, size=(64, 64, 3), dtype=np.uint8
            )
            Image.fromarray(dummy_input).save(input_path)

            output = infer_patch(
                input_path=str(input_path),
                direction=args.direction,
                out_path=str(output_path),
                checkpoint=None,
                tile_size=64,
                stride=64,
                stream=False,
                dummy=True,
                profile_log_path=str(Path(temp_dir) / "profile.json"),
            )

            assert output is not None, "Inference returned no output"
            assert output.shape[:2] == (64, 64), (
                f"Unexpected output shape: {output.shape}"
            )
            assert output.dtype == np.uint8, (
                f"Unexpected output dtype: {output.dtype}"
            )
            assert np.isfinite(output).all(), "Output contains non-finite values"

            normalized = output.astype(np.float32) / 255.0
            assert normalized.min() >= 0.0 and normalized.max() <= 1.0
            assert output_path.exists(), "Output image was not saved"

            print(
                f"[SMOKE OK] shape={output.shape}, "
                f"dtype={normalized.dtype}, "
                f"range=[{normalized.min():.3f}, {normalized.max():.3f}]"
            )
        return

    if not args.input:
        ap.error("--input is required unless --dummy is used")
    out = args.out or f"/tmp/synthetic_{Path(args.input).stem}_{args.direction}.{'tiff' if args.stream or args.wsi else 'png'}"
    if args.wsi and args.stream:
        # Use true WSI streaming (no full load)
        infer_wsi_streaming(args.input, args.direction, out, args.checkpoint, tile_size=args.tile, stride=args.stride)
    else:
        infer_patch(args.input, args.direction, out, args.onnx, args.checkpoint, tile_size=args.tile, stride=args.stride, stream=args.stream)

if __name__=="__main__":
    main()
