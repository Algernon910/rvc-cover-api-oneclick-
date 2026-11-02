#!/usr/bin/env python3
"""One-click installer: local RVC cover backend for AstrBot (Windows + NVIDIA).

What it does
------------
1. clone the upstream RVC FastAPI repo
2. create an isolated venv and install the pinned dependency set
3. install a pre-built fairseq wheel (no MSVC / Cython needed) + py3.12 patches
4. overwrite the files we had to modify upstream (PyAV / librosa / our /cover API)
5. download the three public models (HuBERT / RMVPE / UVR5 separator)
6. write a .env pointing at local asset folders

Usage
-----
    python install.py                      # install into ./runtime
    python install.py --dir D:\\RVC_API     # custom target
    python install.py --skip-deps          # keep existing venv, re-apply patches only
    python install.py --cpu                # no CUDA (very slow, testing only)

Tested with Python 3.12 on Windows 11 + RTX 50 series (CUDA 12.8).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCHES = HERE / "patches"
REPO_URL = "https://github.com/SocAIty/Retrieval-based-Voice-Conversion-FastAPI"
UPSTREAM_DIRNAME = "Retrieval-based-Voice-Conversion-FastAPI"
TORCH_VERSION = "2.7.1"
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
HF = "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main"
HUBERT_URL = f"{HF}/hubert_base.pt"
HUBERT_MIN_SIZE = 150 * 1024 * 1024  # the real file is ~181 MB

# (url, path relative to the upstream repo, minimum plausible size, label)
MODEL_ASSETS = [
    (HUBERT_URL, "assets/hubert/hubert_base.pt", 150 * 1024 * 1024, "HuBERT base (feature extractor)"),
    (f"{HF}/rmvpe.pt", "assets/rmvpe/rmvpe.pt", 150 * 1024 * 1024, "RMVPE (pitch extractor)"),
    (
        f"{HF}/uvr5_weights/HP5_only_main_vocal.pth",
        "assets/uvr5_weights/HP5_only_main_vocal.pth",
        50 * 1024 * 1024,
        "UVR5 HP5 (vocal separator)",
    ),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None, allow_fail: bool = False) -> bool:
    printable = " ".join(str(c) for c in cmd)
    info(f"$ {printable}")
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        if allow_fail:
            info(f"[warn] command failed with exit code {result.returncode}")
            return False
        raise SystemExit(f"\n[ERROR] command failed: {printable}")
    return True


def check_python() -> None:
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise SystemExit(
            f"[ERROR] Python {sys.version_info.major}.{sys.version_info.minor} detected.\n"
            "        This stack needs Python 3.10 - 3.12 (3.12 recommended).\n"
            "        Install it from https://www.python.org/downloads/ and run install again."
        )
    info(f"Python {sys.version.split()[0]} - ok")


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def site_packages(vpy: Path) -> Path:
    # site.getsitepackages()[0] is the venv ROOT on Windows, not Lib\\site-packages
    out = subprocess.check_output(
        [str(vpy), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"]
    )
    return Path(out.decode().strip())


def copy_tree(src: Path, dst: Path) -> int:
    """Copy every file under src into dst, preserving relative paths."""
    count = 0
    for item in src.rglob("*"):
        if item.is_dir() or item.name.startswith("."):
            continue
        target = dst / item.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        info(f"patched {target.relative_to(dst.parent)}")
        count += 1
    return count


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def prepare_dirs(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    updir = root / UPSTREAM_DIRNAME
    if not updir.exists():
        if shutil.which("git"):
            run(["git", "clone", "--depth", "1", REPO_URL, str(updir)])
        else:
            raise SystemExit(
                "[ERROR] git not found in PATH.\n"
                f"        Either install Git, or download the repo zip manually and extract it to:\n"
                f"        {updir}"
            )
    else:
        info(f"upstream already present: {updir}")
    return root, updir


def pip_install(vpy: Path, args: list[str], allow_fail: bool = False) -> bool:
    """pip install with a China mirror first, falling back to pypi.org.

    Plain `pip install` against pypi.org times out constantly on mainland
    networks - that alone can turn a 10 minute install into an hour.
    """
    for index in (PYPI_MIRROR, None):
        cmd = [vpy, "-m", "pip", "install", *args]
        if index:
            cmd += ["--index-url", index]
        if run(cmd, allow_fail=True):
            return True
        info("[warn] retrying with pypi.org")
    if allow_fail:
        return False
    raise SystemExit("[ERROR] pip install failed - check your network / proxy settings")


def install_deps(root: Path, cpu_only: bool) -> Path:
    venv = root / ".venv"
    vpy = venv_python(venv)
    if not vpy.exists():
        info(f"creating venv at {venv}")
        run([sys.executable, "-m", "venv", str(venv)])
        pip_install(vpy, ["--upgrade", "pip", "wheel", "setuptools"])

    # torch first, otherwise later deps may downgrade it to a CPU build
    has_cuda_torch = subprocess.run(
        [str(vpy), "-c", "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True,
    ).returncode == 0
    if has_cuda_torch and not cpu_only:
        info("torch with CUDA already installed - skipping")
    else:
        if cpu_only:
            pip_install(vpy, ["torch", "torchaudio"])
        else:
            ok = run(
                [vpy, "-m", "pip", "install",
                 f"torch=={TORCH_VERSION}+cu128", f"torchaudio=={TORCH_VERSION}+cu128",
                 "--index-url", "https://mirrors.nju.edu.cn/pytorch/whl/cu128"],
                allow_fail=True,
            )
            if not ok:
                info("mirror failed, retrying with the official PyTorch index")
                run([vpy, "-m", "pip", "install",
                     f"torch=={TORCH_VERSION}+cu128", f"torchaudio=={TORCH_VERSION}+cu128",
                     "--index-url", "https://download.pytorch.org/whl/cu128"])

    req = HERE / "requirements-win-py312-cu128.txt"
    pip_install(vpy, ["-r", str(req)])

    # torchcrepe pins librosa==0.9.1; its code only uses librosa.resample (compatible with 0.10)
    pip_install(vpy, ["resampy"])
    pip_install(vpy, ["torchcrepe==0.0.20", "--no-deps"])
    # ffmpeg static binary (used for mixing / loudnorm if none is on PATH)
    pip_install(vpy, ["imageio-ffmpeg"])
    return vpy


def install_fairseq(vpy: Path) -> None:
    wheel = PATCHES / "fairseq-0.12.2-py3-none-any.whl"
    if not wheel.exists():
        raise SystemExit(f"[ERROR] missing {wheel} - re-clone the repo, it ships the prebuilt wheel")
    info("installing prebuilt fairseq (C/Cython extensions already stripped)")
    run([vpy, "-m", "pip", "install", "--no-deps", str(wheel)])

    # --no-deps above is deliberate: letting pip resolve fairseq's metadata would pull
    # omegaconf<2.1 / hydra<1.1, which simply do not work on Python 3.12. These are the
    # versions actually verified to run, so install them explicitly.
    pip_install(vpy, ["omegaconf==2.3.1", "hydra-core==1.3.8", "bitarray", "sacrebleu"])

    sp = site_packages(vpy)
    fairseq_dir = sp / "fairseq"
    if not fairseq_dir.exists():
        raise SystemExit("[ERROR] fairseq not found after install - check the pip output above")

    step("applying fairseq patches (torch 2.6 weights_only / py3.12 dataclass)")
    shutil.copy2(PATCHES / "fairseq/checkpoint_utils.py", fairseq_dir / "checkpoint_utils.py")
    shutil.copy2(PATCHES / "fairseq/dataclass/initialize.py", fairseq_dir / "dataclass/initialize.py")
    shutil.copy2(PATCHES / "fairseq/_rvc_fairseq_py312.py", sp / "_rvc_fairseq_py312.py")
    shutil.copy2(PATCHES / "fairseq/_rvc_fairseq_py312.pth", sp / "_rvc_fairseq_py312.pth")
    info("fairseq patched")


def apply_upstream_patches(updir: Path) -> None:
    step("overwriting upstream files with our fixed versions")
    n = copy_tree(PATCHES / "upstream", updir)
    info(f"{n} files patched")


def download_model(url: str, target: Path, min_size: int, label: str) -> None:
    if target.exists() and target.stat().st_size >= min_size:
        info(f"{label}: already present ({target.stat().st_size // 1024 // 1024} MB)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    step(f"downloading {label} (~{min_size // 1024 // 1024} MB+, hf-mirror)")
    tmp = target.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:  # noqa: BLE001 - network is best effort here
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"[ERROR] download failed: {exc}\n"
            "        Grab the file manually from\n"
            "        https://hf-mirror.com/lj1995/VoiceConversionWebUI/tree/main\n"
            f"        and save it to: {target}"
        )
    if tmp.stat().st_size < min_size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"[ERROR] {label} looks truncated - retry the download")
    tmp.replace(target)
    info(f"saved {target} ({target.stat().st_size // 1024 // 1024} MB)")


def download_models(updir: Path) -> None:
    for url, rel, min_size, label in MODEL_ASSETS:
        download_model(url, updir / rel, min_size, label)


def find_ffmpeg(vpy: Path | None = None) -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"E:\ffmpeg\bin\ffmpeg.exe",
    ):
        if Path(candidate).exists():
            return candidate
    # last resort: the static binary shipped by the imageio-ffmpeg wheel
    if vpy and vpy.exists():
        r = subprocess.run(
            [str(vpy), "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
            capture_output=True,
            text=True,
        )
        exe = r.stdout.strip()
        if r.returncode == 0 and exe and Path(exe).exists():
            return exe
    return ""


def write_env(updir: Path, vpy: Path | None = None) -> None:
    env_file = updir / ".env"
    assets = updir / "assets"
    for sub in ("weights", "uvr5_weights", "indices", "rmvpe"):
        (assets / sub).mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg(vpy)
    if env_file.exists():
        info(".env already exists - left untouched")
        return

    def posix(p: Path) -> str:
        return str(p).replace("\\", "/")

    env_file.write_text(
        "OPENBLAS_NUM_THREADS = 1\n"
        "no_proxy = localhost, 127.0.0.1, ::1\n"
        f"weight_root = {posix(assets / 'weights')}\n"
        f"weight_uvr5_root = {posix(assets / 'uvr5_weights')}\n"
        f"index_root = {posix(assets / 'indices')}\n"
        f"rmvpe_root = {posix(assets / 'rmvpe')}\n"
        f"ffmpeg_path = {posix(Path(ffmpeg)) if ffmpeg else ''}\n",
        encoding="utf-8",
    )
    info(f"wrote {env_file}")
    if not ffmpeg:
        info("[warn] ffmpeg not found in PATH - set ffmpeg_path in .env manually")


def write_launcher(root: Path, updir: Path, vpy: Path) -> Path:
    """Generate a ready-to-use start script with all paths already filled in."""
    launcher = root / "start-backend.bat"
    watcher = HERE / "tools" / "watch_and_open.py"
    content = f"""@echo off
chcp 936 >nul
title RVC Cover Service (127.0.0.1:8000)
setlocal

rem generated by install.py - edit the paths below only if you move the folder
set "PROJECT_DIR={updir}"
set "PYTHON_EXE={vpy}"
set "WATCHER={watcher}"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    pause
    exit /b 1
)

netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo [INFO] Port 8000 is already listening - the service is running.
    start "" http://127.0.0.1:8000/docs
    echo Close the old window first if you want to restart it.
    pause
    exit /b 0
)

cd /d "%PROJECT_DIR%"
set PYTHONPATH=%PROJECT_DIR%;
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8

echo Starting RVC service. First model load takes 20-40 seconds.
echo Docs: http://127.0.0.1:8000/docs
echo KEEP THIS WINDOW OPEN - closing it stops the service.
echo.

if exist "%WATCHER%" start "RVC-ReadyWatcher" /min "%PYTHON_EXE%" "%WATCHER%"

"%PYTHON_EXE%" rvc_fastapi.py

echo.
echo Service exited. Check the messages above for errors.
pause
endlocal
"""
    # GBK + chcp 936: paths may contain non-ASCII characters (e.g. a Chinese user name),
    # and cmd on a zh-CN system reads batch files as cp936.
    launcher.write_bytes(content.replace("\n", "\r\n").encode("gbk", errors="replace"))
    info(f"wrote launcher: {launcher}")
    return launcher


def final_report(root: Path, updir: Path, launcher: Path) -> None:
    assets = updir / "assets"
    print(
        "\n"
        "============================================================\n"
        " Install finished. Remaining manual steps:\n"
        "------------------------------------------------------------\n"
        f" 1. voice models (*.pth)  ->  {assets / 'weights'}\n"
        "    (your own trained voices - the only thing that cannot be shipped)\n"
        f" 2. optional index (*.index) -> {assets / 'indices'}\n"
        "------------------------------------------------------------\n"
        " Then start the backend:\n"
        f"    {launcher}\n"
        "------------------------------------------------------------\n"
        " Finally, in the AstrBot cover plugin set:\n"
        "    rvc_base_url = http://127.0.0.1:8000\n"
        "    rvc_models_keywords = [ \"<your model>.pth|||\" ]\n"
        "============================================================\n",
        flush=True,
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Install the RVC cover backend")
    parser.add_argument("--dir", default=str(HERE / "runtime"), help="install directory")
    parser.add_argument("--skip-deps", action="store_true", help="do not touch the venv / pip")
    parser.add_argument("--skip-models", action="store_true", help="do not download HuBERT / RMVPE / UVR5")
    parser.add_argument("--cpu", action="store_true", help="install CPU-only torch (slow)")
    args = parser.parse_args()

    root = Path(args.dir).expanduser().resolve()

    step("checking environment")
    check_python()
    if shutil.which("git") is None and not (root / UPSTREAM_DIRNAME).exists():
        raise SystemExit("[ERROR] git is required to clone the upstream repo")

    step("preparing directories")
    root, updir = prepare_dirs(root)

    if not args.skip_deps:
        vpy = install_deps(root, args.cpu)
        install_fairseq(vpy)

    apply_upstream_patches(updir)
    if not args.skip_models:
        download_models(updir)
    write_env(updir, venv_python(root / ".venv"))
    launcher = write_launcher(root, updir, venv_python(root / ".venv"))
    final_report(root, updir, launcher)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[cancelled]")
        sys.exit(130)
