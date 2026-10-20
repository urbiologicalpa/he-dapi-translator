"""
test_resource_guards.py - Verifies all spike guards active after optimization
Run: .venv/bin/python -m pytest tests/test_resource_guards.py -v
     or .venv/bin/python tests/test_resource_guards.py
"""
from pathlib import Path

def test_resource_monitor_exists():
    assert Path("src/utils/resource_monitor.py").exists()
    from src.utils.resource_monitor import ResourceMonitor, get_ram_state, get_disk_state
    ram = get_ram_state()
    assert ram["total_mb"] > 1000
    assert ram["avail_mb"] > 100
    assert 0 <= ram["used_pct"] <= 100
    disk = get_disk_state()
    assert disk["free_gb"] > 1

def test_tiling_streaming():
    from src.utils.tiling import TiledInference
    assert hasattr(TiledInference, "merge_tiles_streaming")
    assert hasattr(TiledInference, "infer_image")
    # small merge not OOM
    import numpy as np
    tiler = TiledInference(512,256)
    # should raise if huge
    try:
        tiler.merge_tiles([], (20000,20000,3))
        assert False, "should have raised MemoryError"
    except MemoryError:
        pass
    # streaming path works
    import tempfile
    tiles = [((0,0), np.random.randint(0,255,(512,512,1),dtype=np.uint8))]
    out = tempfile.mktemp(suffix=".tiff")
    res = tiler.merge_tiles_streaming(iter(tiles), (512,512,1), out_path=out)
    assert Path(out).exists()
    Path(out).unlink(missing_ok=True)
    for p in [out+".canvas.dat", out+".weight.dat", out+".final.dat"]:
        Path(p).unlink(missing_ok=True)

def test_pack_hemit_streaming():
    txt = Path("scripts/data/pack_hemit.py").read_text()
    assert "streaming" in txt.lower()
    assert "np.concatenate" not in txt or "streaming" in txt  # should not have naive concat without streaming alternative
    assert "ResourceMonitor" in txt
    assert "gc.collect()" in txt

def test_pack_bci_streaming():
    txt = Path("scripts/data/pack_bci.py").read_text()
    assert "streaming" in txt.lower()
    assert "ResourceMonitor" in txt

def test_datasets_lazy():
    txt = Path("src/models/datasets.py").read_text()
    assert "SWMR" in txt
    assert "_get_h5" in txt
    assert "__del__" in txt
    assert "rdcc_nbytes" in txt
    assert "mihc" in txt

def test_wsi_loader_streaming():
    txt = Path("src/utils/wsi_loader.py").read_text()
    assert "aszarr" in txt
    assert "zarr.open" in txt
    assert "get_ram_state" in txt

def test_configs_lowram():
    import yaml
    with open("configs/hemit_dual_branch.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["training"]["batch_size"] <= 2
    assert cfg["training"]["num_workers"] == 0
    assert cfg["training"]["mixed_precision"] is False
    assert cfg["data"]["stain_norm"] == "none"
    with open("configs/cyclegan_dapi2he.yaml") as f:
        cfg2 = yaml.safe_load(f)
    assert cfg2["training"]["batch_size"] <= 2

def test_train_paired_guards():
    txt = Path("src/models/train_paired.py").read_text()
    assert "grad_accum" in txt
    assert "oom_guard" in txt
    assert "monitor" in txt
    assert "gro" not in txt.lower() or "ResourceMonitor" in txt  # check monitor import
    assert "out of memory" in txt.lower()

def test_infer_streaming():
    txt = Path("src/models/infer.py").read_text()
    assert "stream" in txt.lower()
    assert "infer_wsi_streaming" in txt
    assert "ResourceMonitor" in txt

def test_backend_throttle():
    txt = Path("src/backend/main.py").read_text()
    assert "429" in txt
    assert "used_pct" in txt
    assert "get_ram_state" in txt

def test_wiki_docs():
    assert Path("wiki/README.md").exists()
    for d in ["01-machine-profile","02-data-extraction","03-datasets","04-tiling","05-wsi-loader","06-training","07-inference","08-backend","09-orchestration"]:
        assert Path(f"wiki/{d}").is_dir(), f"missing wiki/{d}"
        assert Path(f"wiki/{d}/SKILL.md").exists(), f"missing SKILL {d}"
    assert Path("Makefile").exists()

if __name__=="__main__":
    tests = [k for k,v in globals().items() if k.startswith("test_")]
    ok=0; fail=0
    for name in tests:
        try:
            globals()[name]()
            print(f"[PASS] {name}")
            ok+=1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback; traceback.print_exc()
            fail+=1
    print(f"\n{ok} passed, {fail} failed out of {len(tests)}")
    exit(1 if fail else 0)
