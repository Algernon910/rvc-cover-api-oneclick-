"""AI 翻唱流水线：音频 → UVR5 人声/伴奏分离 → RVC 变声 → ffmpeg 混音。

被 rvc_fastapi.py 的 /cover 接口调用。
目录结构依赖：必须在项目根目录运行（uvr5 的 modelparams 是相对路径）。
"""

import os
import shutil
import subprocess
import tempfile
import threading
import traceback

import numpy as np
import soundfile as sf
from scipy.io import wavfile

# ffmpeg / ffprobe 可执行文件位置。
# 优先读 .env 的 ffmpeg_path（install.py 会自动写入），其次在 PATH 里找。


def find_ffmpeg() -> str:
    env = os.getenv("ffmpeg_path") or os.getenv("FFMPEG_PATH")
    if env and os.path.exists(env):
        return env.replace("\\", "/")
    which = shutil.which("ffmpeg")
    if which:
        return which.replace("\\", "/")
    return "ffmpeg"


class UVRModelCache:
    """UVR5 分离模型缓存，避免每次请求都重新加载模型（每次约 2 秒）。"""

    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()

    def get(self, model_name: str, agg: int, device: str, is_half: bool):
        key = (model_name, agg, device, is_half)
        with self._lock:
            if key not in self._models:
                from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

                cls = AudioPreDeEcho if ("DeEcho" in model_name or "DeReverb" in model_name) else AudioPre
                weight_root = os.getenv("weight_uvr5_root")
                model_path = os.path.join(weight_root, model_name + ".pth")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"UVR5 模型不存在: {model_path}")
                self._models[key] = cls(
                    agg=int(agg), model_path=model_path, device=device, is_half=is_half
                )
            return self._models[key]


uvr_cache = UVRModelCache()


def normalize_audio(src: str, dst: str, ffmpeg: str = None) -> str:
    """把任意输入音频统一转成 44100Hz 立体声 wav，UVR5 与 RVC 都吃这个格式。"""
    ffmpeg = ffmpeg or find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", src.replace("\\", "/"),
        "-vn", "-ac", "2", "-ar", "44100",
        "-c:a", "pcm_s16le", dst.replace("\\", "/"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError(f"音频转码失败: {proc.stderr[-500:]}")
    return dst


def separate(
    model_name: str,
    src_wav: str,
    work_dir: str,
    agg: int = 10,
    device: str = "cuda:0",
    is_half: bool = True,
) -> tuple:
    """分离出人声与伴奏，返回 (vocal_path, instrumental_path)。"""
    ins_dir = os.path.join(work_dir, "ins")
    voc_dir = os.path.join(work_dir, "voc")
    os.makedirs(ins_dir, exist_ok=True)
    os.makedirs(voc_dir, exist_ok=True)

    pre = uvr_cache.get(model_name, agg, device, is_half)
    pre._path_audio_(src_wav, ins_dir, voc_dir, "wav")

    # 产物命名规则：vocal_<原文件名>_<agg>.wav / instrument_<原文件名>_<agg>.wav
    base = os.path.basename(src_wav)
    vocal = os.path.join(voc_dir, "vocal_%s_%d.wav" % (base, int(agg)))
    inst = os.path.join(ins_dir, "instrument_%s_%d.wav" % (base, int(agg)))
    if not os.path.exists(vocal):
        found = [f for f in os.listdir(voc_dir) if f.lower().startswith("vocal_")]
        if not found:
            raise RuntimeError("UVR5 未产出人声文件")
        vocal = os.path.join(voc_dir, found[0])
    if not os.path.exists(inst):
        found = [f for f in os.listdir(ins_dir) if f.lower().startswith("instrument_")]
        if not found:
            raise RuntimeError("UVR5 未产出伴奏文件")
        inst = os.path.join(ins_dir, found[0])
    return vocal, inst


def convert_vocal(
    vc,
    vocal_path: str,
    out_path: str,
    f0up_key: int = 0,
    f0method: str = "rmvpe",
    index_path: str = None,
    index_rate: float = 0.66,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1,
    protect: float = 0.33,
):
    """用 RVC 把人声转换成目标音色，结果写成 wav 文件。"""
    _, wav_opt = vc.vc_single(
        sid=0,
        input_audio_path=vocal_path,
        f0_up_key=f0up_key,
        f0_file=None,
        f0_method=f0method,
        file_index=index_path,
        file_index2=None,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )
    if wav_opt is None:
        raise RuntimeError("RVC 转换失败")
    wavfile.write(out_path, wav_opt[0], wav_opt[1])
    return out_path


def mix(
    inst_path: str,
    vocal_path: str,
    out_path: str,
    inst_gain: float = 1.0,
    vocal_gain: float = 1.0,
    output_format: str = "mp3",
    normalize: bool = True,
    ffmpeg: str = None,
) -> str:
    """把伴奏与转换后的人声混合成成品。"""
    ffmpeg = ffmpeg or find_ffmpeg()
    # 混音后统一响度，避免转换后人声偏小导致成品整体音量塌陷
    filt = (
        f"[0:a]volume={inst_gain}[a0];[1:a]volume={vocal_gain}[a1];"
        f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=0"
    )
    if normalize:
        filt += ",loudnorm=I=-16:TP=-1.5:LRA=11"
    filt += "[aout]"
    cmd = [
        ffmpeg, "-y",
        "-i", inst_path.replace("\\", "/"),
        "-i", vocal_path.replace("\\", "/"),
        "-filter_complex", filt,
        "-map", "[aout]", "-ac", "2", "-ar", "44100",
    ]
    if output_format == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif output_format == "flac":
        cmd += ["-c:a", "flac"]
    elif output_format == "m4a":
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "pcm_s16le"]
    cmd.append(out_path.replace("\\", "/"))

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"混音失败: {proc.stderr[-500:]}")
    return out_path


def make_cover(
    input_bytes: bytes,
    input_name: str,
    model_name: str,
    vc,
    index_path: str = None,
    f0up_key: int = 0,
    f0method: str = "rmvpe",
    index_rate: float = 0.66,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1,
    protect: float = 0.33,
    uvr_model: str = "HP5_only_main_vocal",
    uvr_agg: int = 10,
    inst_gain: float = 1.0,
    vocal_gain: float = 1.0,
    skip_uvr: bool = False,
    output_format: str = "mp3",
    normalize: bool = True,
    device: str = "cuda:0",
    is_half: bool = True,
    work_dir: str = None,
):
    """完整翻唱流程。返回 (成品路径, work_dir)。由调用方负责清理 work_dir。"""
    own_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="rvc_cover_")
    try:
        src = os.path.join(work_dir, "input_orig" + os.path.splitext(input_name or "")[1] or ".tmp")
        if not os.path.splitext(src)[1]:
            src = os.path.join(work_dir, "input_orig.audio")
        with open(src, "wb") as f:
            f.write(input_bytes)

        norm = os.path.join(work_dir, "input.wav")
        normalize_audio(src, norm)

        if skip_uvr:
            # 输入已经是干声，直接变声，不混伴奏
            vocal_in = norm
            converted = os.path.join(work_dir, "vocal_converted.wav")
            convert_vocal(
                vc, vocal_in, converted,
                f0up_key=f0up_key, f0method=f0method, index_path=index_path,
                index_rate=index_rate, filter_radius=filter_radius,
                resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect,
            )
            out = os.path.join(work_dir, "cover." + output_format)
            if output_format == "wav":
                shutil.copyfile(converted, out)
            else:
                # 单路转码即可，不需要混音
                proc = subprocess.run(
                    [find_ffmpeg(), "-y", "-i", converted.replace("\\", "/"), "-vn"]
                    + (["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"] if normalize else [])
                    + ["-ac", "2", "-ar", "44100"]
                    + (["-c:a", "libmp3lame", "-b:a", "192k"] if output_format == "mp3"
                     else ["-c:a", "flac"] if output_format == "flac"
                     else ["-c:a", "aac", "-b:a", "192k"] if output_format == "m4a"
                     else ["-c:a", "pcm_s16le"]) +
                    [out.replace("\\", "/")],
                    capture_output=True, text=True, encoding="utf-8", errors="ignore",
                )
                if not os.path.exists(out) or os.path.getsize(out) == 0:
                    raise RuntimeError(f"转码失败: {proc.stderr[-500:]}")
            return out, work_dir

        vocal, inst = separate(uvr_model, norm, work_dir, agg=uvr_agg,
                               device=device, is_half=is_half)

        converted = os.path.join(work_dir, "vocal_converted.wav")
        convert_vocal(
            vc, vocal, converted,
            f0up_key=f0up_key, f0method=f0method, index_path=index_path,
            index_rate=index_rate, filter_radius=filter_radius,
            resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect,
        )

        out = os.path.join(work_dir, "cover." + output_format)
        mix(inst, converted, out, inst_gain=inst_gain, vocal_gain=vocal_gain,
            output_format=output_format, normalize=normalize)
        return out, work_dir
    except Exception:
        if own_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise


def cleanup(work_dir: str):
    shutil.rmtree(work_dir, ignore_errors=True)
