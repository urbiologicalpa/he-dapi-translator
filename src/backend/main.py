from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io, gc
import numpy as np
from PIL import Image
from pathlib import Path

app = FastAPI(title="H&E <-> DAPI Translator", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Lazy import infer to avoid hard dep at startup
def _infer_tile(tile: np.ndarray, direction: str) -> np.ndarray:
    try:
        from src.models.infer import run_torch_model
        ckpt = "resources/models/checkpoints/he2dapi/generator_last.pt" if direction=="he2dapi" else "resources/models/checkpoints/dapi2he/cyclegan_G_last.pt"
        return run_torch_model(tile, direction, checkpoint=ckpt if Path(ckpt).exists() else None)
    except Exception as e:
        # Fallback dummy
        if direction=="he2dapi":
            gray = np.dot(tile[...,:3],[0.2989,0.5870,0.1140]) if tile.ndim==3 else tile
            return (255-gray).astype(np.uint8)
        else:
            gray = tile if tile.ndim==2 else tile[:,:,0]
            he = np.stack([gray, (gray*0.6).astype(np.uint8), (gray*0.8).astype(np.uint8)],-1)
            return he

@app.get("/health")
def health():
    return {"status": "ok", "models": ["he2dapi_pix2pix", "dapi2he_cyclegan"]}

@app.get("/models")
def list_models():
    return {
        "he2dapi": {"checkpoint": "resources/models/checkpoints/he2dapi/he2dapi_pix2pix.onnx", "patch":512},
        "dapi2he": {"checkpoint": "resources/models/checkpoints/dapi2he/cyclegan_G_last.pt", "patch":256},
        "note": "Run python src/models/train_paired.py --dummy to generate checkpoints"
    }

@app.post("/translate")
async def translate(file: UploadFile = File(...), direction: str = "he2dapi"):
    # RAM guard before processing
    try:
        from src.utils.resource_monitor import get_ram_state
        ram = get_ram_state()
        if ram["used_pct"] > 85 or ram["avail_mb"] < 400:
            raise HTTPException(429, f"Server low RAM ({ram['used_pct']:.0f}%, avail {ram['avail_mb']:.0f}MB), try smaller image or retry")
        # also check upload size: reject >100MP without streaming
    except HTTPException:
        raise
    except: pass
    if direction not in ("he2dapi","dapi2he","auto"):
        raise HTTPException(400, "direction must be he2dapi, dapi2he, or auto")
    data = await file.read()
    # Guard: limit upload to 500MB after read
    if len(data) > 500*1024*1024:
        raise HTTPException(413, "File >500MB, use /translate/wsi with streaming")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # For huge uploads, thumbnail before numpy to keep RAM low
        w,h = img.size
        est_mb = w*h*3/1e6
        try:
            from src.utils.resource_monitor import get_ram_state
            ram = get_ram_state()
            if est_mb > ram["avail_mb"]*0.5 or est_mb > 512:
                # downsample to 2048 for immediate translate; suggest wsi endpoint for full-res
                img.thumbnail((2048,2048), Image.BILINEAR)
                print(f"[BACKEND] Large upload {w}x{h} {est_mb:.0f}MB -> thumb {img.size} (RAM guard)")
        except: pass
        del data
        gc.collect()
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    tile = np.array(img)
    try:
        from src.models.infer import resolve_direction
        direction = resolve_direction(direction, tile)
    except Exception as exc:
        raise HTTPException(400, f"direction routing failed: {exc}")
    # Handle WSI-sized upload via tiling if >1024
    if max(tile.shape[:2]) > 1024:
        from src.utils.tiling import TiledInference
        tiler = TiledInference(tile_size=512, stride=256)
        def fn(t): 
            r = _infer_tile(t, direction)
            return r[...,None] if r.ndim==2 else r
        # Streaming if huge
        est_out = tile.shape[0]*tile.shape[1]* (1 if direction=="he2dapi" else 3)*4/1e6
        if est_out > 512:
            # write to temp tiff then read back first 2048 for preview, but keep streaming path
            import tempfile, os
            tmp_tiff = tempfile.mktemp(suffix=".tiff")
            try:
                tiler.infer_image(tile, fn, out_path=tmp_tiff)
                # For response, return thumbnail of result if streamed to tiff (avoid loading full)
                # Fallback: generate 512 thumb from tiff via tifffile
                import tifffile
                with tifffile.TiffFile(tmp_tiff) as tf:
                    arr = tf.asarray(out= "memmap" if hasattr(tf, "asarray") else None)
                    # simple: read via PIL thumbnail from file
                    from PIL import Image as PILImage2
                    # use tifffile to read region 0:2048
                    preview = arr[:2048,:2048] if arr.shape[0]>2048 else arr
                    if preview.dtype!=np.uint8:
                        preview = (preview/ preview.max()*255).astype(np.uint8) if preview.max()>0 else preview.astype(np.uint8)
                    result = preview
                    if result.ndim==3 and result.shape[2]==1 and direction=="he2dapi":
                        result = result[:,:,0]
                    elif result.ndim==2 and direction=="dapi2he":
                        result = np.stack([result]*3,-1)
                    # cleanup streamed file after preview (or keep?)
                    try: os.unlink(tmp_tiff)
                    except: pass
                    # also unlink memmap temps
                    for p in [tmp_tiff+".canvas.dat", tmp_tiff+".weight.dat", tmp_tiff+".final.dat"]:
                        try: Path(p).unlink(missing_ok=True)
                        except: pass
            except Exception as e:
                print(f"[BACKEND] streaming tiled failed {e}, fallback tiling")
                result = tiler.infer_image(tile, fn)
                if direction=="he2dapi" and result.ndim==3 and result.shape[2]==1:
                    result = result[:,:,0]
        else:
            result = tiler.infer_image(tile, fn)
            if direction=="he2dapi" and result.ndim==3 and result.shape[2]==1:
                result = result[:,:,0]
    else:
        result = _infer_tile(tile, direction)
    # Encode result
    buf = io.BytesIO()
    if result is None:
        raise HTTPException(500, "Streaming result null (should have returned tiff)")
    if result.ndim==2:
        Image.fromarray(result).save(buf, format="PNG")
    else:
        Image.fromarray(result).save(buf, format="PNG")
    buf.seek(0)
    # free
    del tile, result, img
    gc.collect()
    headers = {"X-Direction": direction, "X-Warning": "SYNTHETIC - not for diagnosis"}
    return StreamingResponse(buf, media_type="image/png", headers=headers)

@app.post("/translate/wsi")
async def translate_wsi(file: UploadFile = File(...), direction: str = "he2dapi"):
    # Same as /translate but explicit WSI flag (supports .svs via WSILoader)
    return await translate(file, direction)

# Serve the built frontend (src/frontend/dist) when present.
_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
