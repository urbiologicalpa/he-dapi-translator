import zipfile, os, time, sys
zip_path="data/raw/hemit/hemit.zip"
out="data/raw/hemit/HEMIT.7z"
tmp_root="data/raw/hemit/tmp_unzip"
print(f"[{time.ctime()}] start extract outer zip", flush=True)
with zipfile.ZipFile(zip_path) as z:
    name=z.namelist()[0]
    info=z.getinfo(name)
    print(f"inner {name} file_size={info.file_size} compress={info.compress_size}", flush=True)
    z.extract(name, path=tmp_root)
    src = os.path.join(tmp_root, name)
    sz=os.path.getsize(src)
    print(f"extracted tmp size {sz}", flush=True)
    os.replace(src, out)
    print(f"moved to {out} size {os.path.getsize(out)}", flush=True)
    import shutil
    shutil.rmtree(tmp_root, ignore_errors=True)
print(f"[{time.ctime()}] outer done", flush=True)
import py7zr
seven="data/raw/hemit/HEMIT.7z"
outdir="data/raw/hemit/extracted"
os.makedirs(outdir, exist_ok=True)
print(f"[{time.ctime()}] start 7z extract", flush=True)
with py7zr.SevenZipFile(seven, mode='r') as z:
    files=z.list()
    print(f"7z entries {len(files)}", flush=True)
    for f in files[:5]:
        print(f.filename, f.uncompressed, flush=True)
    z.extractall(path=outdir)
print(f"[{time.ctime()}] 7z done", flush=True)
import subprocess
subprocess.run(["du","-sh",outdir])
subprocess.run(["ls","-R",outdir], stdout=open("/tmp/hemit_final_ls.txt","w"))
print("listed", flush=True)
