
from typing import Union
import os
import re
import sys
from io import BytesIO

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# don't like settings paths like this at all but due bad code its necessary
now_dir = os.getcwd()
sys.path.append(now_dir)
print(now_dir)

from dotenv import load_dotenv
from scipy.io import wavfile

from configs.config import Config
from infer.modules.vc.modules import VC
from cover_pipeline import make_cover, cleanup
import fastapi
import uvicorn

# load_dotenv is also very bad practice but necessary due bad code
load_dotenv()

app = fastapi.FastAPI(
    title="Retrieval-based Voice Conversion FastAPI",
    summary="Infer previously cloned voices",
    version="0.0.2",
    contact={
        "name": "w4hns1nn",
        "url": "https://github.com/w4hns1nn",
    }
)

tags = [
    {
        "name": "voice2voice",
        "description": "Voice2Voice conversion using the pretrained model"
    }
]

class ModelCache:
    """
    This class is used to cache the models so that they don't need to be loaded every time
    """
    def __init__(self):
        self.models = {}

    def load_model(self, model_name: str, device: str = None, is_half: bool = True):
        if model_name not in self.models:
            config = Config() # config_file_folder="A:/projects/Retrieval-based-Voice-Conversion-WebUI/configs/")
            config.device = device if device else config.device
            config.is_half = is_half if is_half else config.is_half
            vc = VC(config)
            vc.get_vc(f"{model_name}.pth")
            self.models[model_name] = vc
        return self.models[model_name]


def resolve_index_path(model_name: str) -> str:
    """在 .env 的 index_root 下找到该音色对应的 .index 文件。

    原实现只认写死的 logs/<模型名>/added_IVF1254_Flat_nprobe_1_<模型名>_v2.index，
    但实际训练产物的 IVF 值可能是 371，且 .pth 文件名带 _e<轮次>_s<步数> 后缀
    （如 my_voice_e180_s4320.pth）。这里按"去掉训练后缀的基础名"做模糊匹配。
    """
    index_root = os.getenv("index_root") or "logs"
    base = re.sub(r"_e\d+_s\d+$", "", model_name)

    fuzzy = []
    if os.path.isdir(index_root):
        for root, _, files in os.walk(index_root, topdown=False):
            for name in files:
                if not name.endswith(".index") or "trained" in name:
                    continue
                full = os.path.join(root, name)
                if base and base in name:
                    return full
                if model_name in name or (base and base in root):
                    fuzzy.append(full)

    default_path = os.path.join(
        "logs", model_name, f"added_IVF1254_Flat_nprobe_1_{model_name}_v2.index"
    )
    if os.path.exists(default_path):
        return default_path
    return fuzzy[0] if fuzzy else None


def infer(
        input: Union[str, bytes], # filepath or raw bytes
        model_name: str,
        index_path: str = None,
        f0up_key: int = 0,
        f0method: str = "crepe",
        index_rate: float = 0.66,
        device: str = None,
        is_half: bool = False,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 1,
        protect: float = 0.33,
        **kwargs
    ):
    model_name = model_name.replace(".pth", "")

    if index_path is None:
        index_path = resolve_index_path(model_name)
        if index_path is None:
            raise ValueError(
                f"在 {os.getenv('index_root') or 'logs'} 下没有找到 {model_name} 对应的 .index 文件，"
                f"请显式传入 index_path"
            )

    vc = model_cache.load_model(model_name, device=device, is_half=is_half)

    _, wav_opt = vc.vc_single(
        sid=0,
        input_audio_path=input,
        f0_up_key=f0up_key,
        f0_file=None,
        f0_method=f0method,
        file_index=index_path,
        file_index2=None,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect
    )

    # using virtual file to be able to return it as response
    wf = BytesIO()
    wavfile.write(wf, wav_opt[0], wav_opt[1])
    return wf


@app.post("/voice2voice", tags=["voice2voice"])
async def voice2voice(
    input_file: fastapi.UploadFile,
    model_name: str,
    index_path: str = None,
    f0up_key: int = 0,
    f0method: str = "crepe",
    index_rate: float = 0.66,
    device: str = None,
    is_half: bool = False,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1,
    protect: float = 0.33
):
    """
    :param input_file: the .wav file to be converted
    :param model_name: the name of the model which was previously trained in the gui like the name in logs folder
    :param index_path: the index file of the previously trained model if none then use default dir logs by rvc.
    :param f0up_key: 0 or 1
    :param f0method: harvest, pm, crepe or rmvpe
    :param index_rate: 0.66
    :param device: if none then use default by rvc. cuda or cpu or specific cuda device "cuda:0", "cuda:1"
    :param is_half: False or True
    :param filter_radius: 3
    :param resample_sr: 0
    :param rms_mix_rate: 1
    :param protect: 0.33
    """
    audio_bytes = await input_file.read()

    kwargs = locals()
    kwargs["input"] = audio_bytes
    del kwargs["input_file"]

    # call the infer function
    try:
        wf = infer(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(wf, media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename=rvc.wav"})


@app.post("/voice2voice_local", tags=["voice2voice"])
async def voice2voice_local(
    input_path: str,
    model_name: str,
    index_path: str = None,
    opt_path: str = None,
    f0up_key: int = 0,
    f0method: str = "crepe",
    index_rate: float = 0.66,
    device: str = None,
    is_half: bool = False,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1,
    protect: float = 0.33
):
    """
    :param input_path: the .wav file to be converted
    :param model_name: the name of the model which was previously trained in the gui like the name in logs folder
    :param index_path: the index file of the previously trained model if none then use default dir logs by rvc.
    :param opt_path: if not None then save the result to this path
    :param f0up_key: 0 or 1
    :param f0method: harvest, pm, crepe or rmvpe
    :param index_rate: 0.66
    :param device: if none then use default by rvc. cuda or cpu or specific cuda device "cuda:0", "cuda:1"
    :param is_half: False or True
    :param filter_radius: 3
    :param resample_sr: 0
    :param rms_mix_rate: 1
    :param protect: 0.33
    """
    kwargs = locals()

    # call the infer function
    try:
        wf = infer(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # write to file if opt_path is provided
    if opt_path is not None:
        out_name = os.path.basename(opt_path)
        with open(opt_path, "wb") as f:
            f.write(wf.getbuffer())
    else:
        out_name = os.path.basename("input_path")

    return StreamingResponse(wf, media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename={out_name}"})

@app.get("/status")
def status():
    return {"status": "ok"}


@app.get("/models")
def models():
    """列出 weight_root 下可用的音色模型名（不含 .pth 后缀）。"""
    root = os.getenv("weight_root") or "assets/weights"
    names = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            if name.lower().endswith(".pth"):
                names.append(name[:-4])
    names.sort()
    return {"models": names, "weight_root": root}


@app.post("/cover", tags=["cover"])
async def cover(
    input_file: fastapi.UploadFile,
    model_name: str,
    index_path: str = None,
    f0up_key: int = 0,
    f0method: str = "rmvpe",
    index_rate: float = 0.66,
    device: str = None,
    is_half: bool = True,
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
    normalize: bool = True
):
    """AI 翻唱：分离人声/伴奏 → 目标音色转换 → 混音。

    :param input_file: 歌曲音频（mp3/wav/flac/m4a 等）
    :param model_name: 音色模型名，如 my_voice_e180_s4320
    :param uvr_model: 分离模型，HP5_only_main_vocal / HP2_all_vocals / VR-DeEchoNormal…
    :param skip_uvr: 输入已是干声（无人伴奏）时设为 true，只做变声不混音
    :param output_format: 输出格式 mp3 / wav / flac / m4a
    :param vocal_gain: 转换后人声音量倍数
    :param inst_gain: 伴奏音量倍数
    :param normalize: 是否做响度归一化（统一到 -16 LUFS，成品音量更稳）
    """
    import time as _time

    t0 = _time.time()
    audio_bytes = await input_file.read()
    model_name = model_name.replace(".pth", "")

    resolved_device = device or _default_device()
    resolved_index = index_path or resolve_index_path(model_name)

    try:
        vc = model_cache.load_model(model_name, device=resolved_device, is_half=is_half)
        out_path, work_dir = make_cover(
            audio_bytes,
            input_file.filename or "input.mp3",
            model_name,
            vc,
            index_path=resolved_index,
            f0up_key=f0up_key,
            f0method=f0method,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            uvr_model=uvr_model,
            uvr_agg=uvr_agg,
            inst_gain=inst_gain,
            vocal_gain=vocal_gain,
            skip_uvr=skip_uvr,
            output_format=output_format,
            normalize=normalize,
            device=resolved_device,
            is_half=is_half,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    ext = output_format.lower()
    media_type = {
        "mp3": "audio/mpeg", "wav": "audio/wav",
        "flac": "audio/flac", "m4a": "audio/mp4",
    }.get(ext, "application/octet-stream")

    file_size = os.path.getsize(out_path)
    elapsed = _time.time() - t0
    print(f"[cover] {model_name} 完成 | 产物 {file_size / 1024 / 1024:.2f}MB | 耗时 {elapsed:.1f}s")

    def _iterfile():
        with open(out_path, "rb") as f:
            yield from f
        cleanup(work_dir)

    return StreamingResponse(
        _iterfile(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename=cover.{ext}',
            "X-Elapsed-Seconds": f"{elapsed:.2f}",
        },
    )


def _default_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# create model cache
model_cache = ModelCache()

# Gradio 协议兼容层：让 matsuko_cover 等走 gradio_client 的插件可以直接对接本服务
import gradio_shim
gradio_shim.register(app, app_ctx={
    "model_cache": model_cache,
    "resolve_index": resolve_index_path,
    "default_device": _default_device,
})

# start uvicorn server
# 默认监听 127.0.0.1:8000，可用环境变量 RVC_HOST / RVC_PORT 覆盖
uvicorn.run(
    app,
    host=os.getenv("RVC_HOST", "127.0.0.1"),
    port=int(os.getenv("RVC_PORT", "8000")),
)

