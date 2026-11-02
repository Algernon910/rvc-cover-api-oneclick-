"""Gradio 协议兼容层（shim）。

让 AstrBot 的 matsuko_cover 插件（走 gradio_client 协议，api_name=/show_model、/convert）
能直接把本 FastAPI 服务当作 RVCSVC-API 后端使用。

对外暴露的 Gradio 协议端点：
  GET  /config                  → 伪 Gradio app 配置
  GET  /info?serialize=False    → API 描述（参数名/顺序）
  POST /upload                  → 文件上传（插件传本地路径时用不到，兜底）
  POST /queue/data              → 提交任务，返回 event_id
  GET  /queue/data?session_hash → SSE 事件流
  POST /reset                   → 取消任务（占位）
  GET  /heartbeat/{session}     → SSE 心跳
  GET  /file={path}             → 产物文件下载

/convert 语义按 RVCSVC-API 的参数契约映射到本服务的 cover 流水线：
  song_name_src: 本地路径 / http(s) URL / 网易云歌曲 ID 或歌名
  model_dropdown: 音色模型名；key_shift: 变调半音数；f0_method/index_rate/filter_radius 透传；
  uvr5_agg/uvr5_tta 透传给 UVR5；vocal_vol/inst_vol 按 dB 映射为混音增益；
  其余（reverb/delay/msst/vocal_postprocess…）本后端不支持，接收但忽略。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
import time
import traceback
import uuid

import fastapi
import fastapi.responses
import httpx

from cover_pipeline import (
    cleanup,
    convert_vocal,
    find_ffmpeg,
    mix,
    normalize_audio,
    separate,
    uvr_cache,
)

# ---------------------------------------------------------------- 常量

CONVERT_PARAMS = [
    "song_name_src",
    "key_shift",
    "vocal_vol",
    "inst_vol",
    "model_dropdown",
    "reverb_intensity",
    "delay_intensity",
    "uvr5_agg",
    "uvr5_tta",
    "uvr5_postprocess",
    "uvr5_window_size",
    "uvr5_high_end_process",
    "msst_batch_size",
    "msst_num_overlap",
    "msst_normalize",
    "vocal_postprocess",
    "shift_accompaniment",
    "f0_method",
    "index_rate",
    "filter_radius",
]

NETEASE_WEB_API = "http://music.163.com"  # 官方接口（搜索）
METING_API_BASE = (
    os.getenv("meting_api_base") or "https://api.injahow.cn/meting"
).rstrip("/")  # 取流代理（实测可直出音频）
COVER_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cover_out")
UPLOAD_DIR = os.path.join(COVER_OUT_DIR, "uploads")
os.makedirs(COVER_OUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 同一时刻只跑一个推理任务（单卡）
gpu_lock = threading.Lock()

# session_hash -> {"buffer": [msg...], "cond": asyncio.Condition, "closed": bool}
_sessions: dict = {}
_sessions_lock = threading.Lock()


def _session(session_hash: str) -> dict:
    with _sessions_lock:
        s = _sessions.get(session_hash)
        if s is None:
            s = {"buffer": [], "cond": asyncio.Condition(), "closed": False}
            _sessions[session_hash] = s
        return s


async def _push(session_hash: str, msg: dict):
    s = _session(session_hash)
    async with s["cond"]:
        s["buffer"].append(msg)
        s["cond"].notify_all()


# ---------------------------------------------------------------- 歌曲解析


async def resolve_song(src: str, progress=None) -> str:
    """把 song_name_src 解析成本地音频文件路径。

    支持本地路径、http(s) URL、网易云歌曲 ID（纯数字）或歌名（自动搜索）。
    """
    src = (src or "").strip()
    if not src:
        raise ValueError("song_name_src 为空")

    if os.path.isfile(src):
        return src

    async def _report(text):
        if progress:
            await progress(text)

    if re.match(r"^https?://", src, re.I):
        await _report("正在下载音频链接")
        return await _download_to_file(src, "dl")

    if src.isdigit():
        await _report(f"正在获取网易云歌曲 {src}")
        return await _download_to_file(_meting_url(src), "netease_" + src)

    # 当作歌名搜索（网易云官方接口）
    await _report(f"正在网易云搜索《{src}》")
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.post(
            NETEASE_WEB_API + "/api/search/get/web",
            data={"s": src, "limit": 5, "type": 1, "offset": 0},
            headers={"Referer": "http://music.163.com"},
        )
        songs = ((r.json() or {}).get("result") or {}).get("songs") or []
        if not songs:
            raise ValueError(f"网易云没有搜到《{src}》")
        song = songs[0]
        await _report(f"正在下载《{song['name']}》")
    return await _download_to_file(_meting_url(song["id"]), "netease_%s" % song["id"])


def _meting_url(song_id) -> str:
    """Meting 代理取流地址，直接返回音频流（实测可用）。"""
    return "%s/?server=netease&type=url&id=%s" % (METING_API_BASE, song_id)


async def _download_to_file(url: str, tag: str) -> str:
    ext = ".mp3"
    m = re.search(r"\.(mp3|flac|m4a|wav|ogg|aac)(?:[?#]|$)", url, re.I)
    if m:
        ext = "." + m.group(1).lower()
    dest = os.path.join(UPLOAD_DIR, "%s_%s%s" % (tag, uuid.uuid4().hex[:10], ext))
    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as cli:
        async with cli.stream("GET", url) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in r.aiter_bytes(65536):
                    f.write(chunk)
    if os.path.getsize(dest) < 1024:
        raise ValueError("下载的音频文件异常（过小）")
    return dest


# ---------------------------------------------------------------- 任务处理


def _list_models() -> list:
    root = os.getenv("weight_root") or "assets/weights"
    if not os.path.isdir(root):
        return []
    return sorted(
        n for n in os.listdir(root) if n.lower().endswith(".pth")
    )


def _db_gain(vol) -> float:
    """matsuko 的音量参数按 dB 语义映射为线性增益，0 表示不变。"""
    try:
        v = float(vol or 0)
    except (TypeError, ValueError):
        return 1.0
    return round(10 ** (v / 20), 4) if v else 1.0


def _run_convert(args: dict, model_cache, resolved_index_for, default_device) -> str:
    """阻塞的翻唱流水线，返回成品文件路径。"""
    src = args["song_name_src"]
    model_name = str(args.get("model_dropdown") or "").strip()
    model_name = model_name.split("|||")[0].replace(".pth", "").strip()
    if not model_name:
        raise ValueError("未指定音色模型")

    f0up_key = int(float(args.get("key_shift") or 0))
    f0method = str(args.get("f0_method") or "rmvpe").strip().lower()
    index_rate = float(args.get("index_rate") or 0.75)
    filter_radius = int(float(args.get("filter_radius") or 3))
    uvr_agg = int(float(args.get("uvr5_agg") or 10))
    uvr_tta = bool(args.get("uvr5_tta"))
    vocal_gain = _db_gain(args.get("vocal_vol"))
    inst_gain = _db_gain(args.get("inst_vol"))
    index_path = resolved_index_for(model_name)
    device = default_device()

    work_dir = tempfile.mkdtemp(prefix="rvc_cover_")
    try:
        norm = os.path.join(work_dir, "input.wav")
        normalize_audio(src, norm)

        with gpu_lock:
            vocal, inst = separate("HP5_only_main_vocal", norm, work_dir,
                                   agg=uvr_agg, device=device, is_half=True)
            yield_progress = args.get("_progress")
            converted = os.path.join(work_dir, "vocal_converted.wav")
            vc = model_cache.load_model(model_name, device=device, is_half=True)
            convert_vocal(
                vc, vocal, converted,
                f0up_key=f0up_key, f0method=f0method, index_path=index_path,
                index_rate=index_rate, filter_radius=filter_radius,
            )
            out_dir = os.path.join(COVER_OUT_DIR, uuid.uuid4().hex[:12])
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, "cover.mp3")
            mix(inst, converted, out, inst_gain=inst_gain, vocal_gain=vocal_gain,
                output_format="mp3")
        return out
    finally:
        pass


async def _job_convert(session: str, event_id: str, data: list, app_ctx: dict):
    args = dict(zip(CONVERT_PARAMS, data))

    async def progress(text):
        await _push(session, {
            "msg": "progress", "event_id": event_id, "rank": 0,
            "rank_eta": None, "queue_size": 0, "success": None,
            "progress_data": [{"index": None, "length": None, "unit": "steps",
                               "progress": None, "desc": text}],
        })

    loop = asyncio.get_running_loop()
    t0 = time.time()
    stage = {"name": ""}

    async def report(text):
        stage["name"] = text
        await progress(text)

    # 把进度回调传入阻塞函数：通过轮询 stage 不现实，这里直接把 report 的同步代理塞进 args
    args["_progress"] = None

    try:
        src = await resolve_song(str(args.get("song_name_src") or ""), progress=report)
        args["song_name_src"] = src

        await report("RVC 音色转换中（%s）" % args.get("model_dropdown", ""))
        out = await loop.run_in_executor(
            None, _run_convert, args, app_ctx["model_cache"],
            app_ctx["resolve_index"], app_ctx["default_device"],
        )
        elapsed = time.time() - t0
        print(f"[gradio-shim] convert 完成 | {os.path.basename(out)} | 耗时 {elapsed:.1f}s")
        await _push(session, {
            "msg": "process_completed", "event_id": event_id,
            "success": True, "rank": 0, "rank_eta": elapsed, "queue_size": 0,
            "progress_data": [],
            "output": {"data": [
                {"path": out.replace("\\", "/"), "orig_name": "cover.mp3",
                 "meta": {"_type": "gradio.FileData"}},
                "false",
            ]},
        })
    except Exception as e:
        traceback.print_exc()
        await _push(session, {
            "msg": "process_completed", "event_id": event_id,
            "success": False, "rank": 0, "rank_eta": None, "queue_size": 0,
            "progress_data": [], "output": {"data": []},
            "error": f"{type(e).__name__}: {e}",
        })


async def _run_job(session_hash: str, fn_index: int, data: list, app_ctx: dict, event_id: str = None):
    event_id = event_id or uuid.uuid4().hex[:16]
    try:
        await _push(session_hash, {
            "msg": "process_starts", "event_id": event_id, "rank": 0,
            "rank_eta": None, "queue_size": 0, "success": None, "progress_data": [],
        })
        if fn_index == 0:  # /show_model
            await _push(session_hash, {
                "msg": "process_completed", "event_id": event_id, "success": True,
                "rank": 0, "rank_eta": None, "queue_size": 0, "progress_data": [],
                "output": {"data": [_list_models()]},
            })
        elif fn_index == 1:  # /convert
            asyncio.get_running_loop().create_task(
                _job_convert(session_hash, event_id, data, app_ctx)
            )
        elif fn_index == 2:  # /cache_info
            await _push(session_hash, {
                "msg": "process_completed", "event_id": event_id, "success": True,
                "rank": 0, "rank_eta": None, "queue_size": 0, "progress_data": [],
                "output": {"data": [json.dumps({"total": 0, "items": []}, ensure_ascii=False)]},
            })
        elif fn_index == 3:  # /clear_cache
            await _push(session_hash, {
                "msg": "process_completed", "event_id": event_id, "success": True,
                "rank": 0, "rank_eta": None, "queue_size": 0, "progress_data": [],
                "output": {"data": ["缓存已清理"]},
            })
        else:
            raise ValueError(f"未知 fn_index {fn_index}")
    except Exception as e:
        traceback.print_exc()
        await _push(session_hash, {
            "msg": "process_completed", "event_id": event_id, "success": False,
            "rank": 0, "rank_eta": None, "queue_size": 0, "progress_data": [],
            "output": {"data": []}, "error": f"{type(e).__name__}: {e}",
        })


# ---------------------------------------------------------------- 协议端点


def _param(name: str) -> dict:
    return {
        "parameter_name": name,
        "parameter_has_default": True,
        "parameter_default": None,
        "type": "Any",
        "python_type": {"type": "Any"},
        "component": "Textbox",
        "example_input": None,
    }


def _build_config() -> dict:
    components = []
    deps = []
    # show_model: outputs [1]
    deps.append({"id": 0, "api_name": "show_model", "inputs": [],
                 "outputs": [1], "backend_fn": True})
    # convert: 20 inputs + 2 outputs
    deps.append({"id": 1, "api_name": "convert",
                 "inputs": list(range(10, 10 + len(CONVERT_PARAMS))),
                 "outputs": [30, 31], "backend_fn": True})
    deps.append({"id": 2, "api_name": "cache_info", "inputs": [],
                 "outputs": [32], "backend_fn": True})
    deps.append({"id": 3, "api_name": "clear_cache", "inputs": [],
                 "outputs": [33], "backend_fn": True})
    for cid in list(range(10, 10 + len(CONVERT_PARAMS))) + [1, 30, 31, 32, 33]:
        components.append({"id": cid, "type": "textbox", "skip_api": False})
    return {
        "version": "4.44.1",
        "mode": "api",
        "api_prefix": "",
        "protocol": "sse_v1",
        "components": components,
        "dependencies": deps,
        "max_file_size": None,
    }


def _build_info() -> dict:
    def ep(params):
        return {"parameters": params, "returns": []}

    return {
        "named_endpoints": {
            "/show_model": ep([]),
            "/convert": ep([_param(p) for p in CONVERT_PARAMS]),
            "/cache_info": ep([]),
            "/clear_cache": ep([]),
        },
        "unnamed_endpoints": {},
    }


def register(app: fastapi.FastAPI, app_ctx: dict):
    """把 Gradio 协议端点挂到 FastAPI 应用上。

    app_ctx 需包含：model_cache（模型缓存对象）、resolve_index（函数）、
    default_device（函数）。
    """

    @app.get("/config", include_in_schema=False)
    def gradio_config():
        return _build_config()

    @app.get("/info", include_in_schema=False)
    def gradio_info():
        return _build_info()

    @app.post("/upload", include_in_schema=False)
    async def gradio_upload(request: fastapi.Request):
        form = await request.form()
        saved = []
        for _, file in form.multi_items():
            if hasattr(file, "read"):
                dest = os.path.join(UPLOAD_DIR, uuid.uuid4().hex[:12] + "_" + (file.filename or "file"))
                with open(dest, "wb") as f:
                    f.write(await file.read())
                saved.append(dest.replace("\\", "/"))
        return saved

    @app.post("/queue/join", include_in_schema=False)
    @app.post("/queue/data", include_in_schema=False)
    async def gradio_queue_join(request: fastapi.Request):
        body = await request.json()
        session_hash = body.get("session_hash") or uuid.uuid4().hex
        fn_index = int(body.get("fn_index") or 0)
        data = body.get("data") or []
        _session(session_hash)  # 确保会话存在
        event_id = uuid.uuid4().hex[:16]
        asyncio.get_running_loop().create_task(
            _run_job(session_hash, fn_index, data, app_ctx, event_id)
        )
        return {"event_id": event_id}

    @app.get("/queue/data", include_in_schema=False)
    async def gradio_queue_stream(request: fastapi.Request):
        session_hash = request.query_params.get("session_hash") or ""
        s = _session(session_hash)

        async def stream():
            idx = 0
            last_beat = time.time()
            try:
                while True:
                    async with s["cond"]:
                        if idx >= len(s["buffer"]):
                            try:
                                await asyncio.wait_for(s["cond"].wait(), timeout=1.0)
                            except asyncio.TimeoutError:
                                pass
                        msgs = s["buffer"][idx:]
                        idx = len(s["buffer"])
                    now = time.time()
                    if msgs:
                        for m in msgs:
                            yield "data: " + json.dumps(m, ensure_ascii=False) + "\n\n"
                        last_beat = now
                    elif now - last_beat > 15:
                        yield "data: " + json.dumps({"msg": "heartbeat"}) + "\n\n"
                        last_beat = now
                    else:
                        await asyncio.sleep(0.05)
            finally:
                pass

        return fastapi.responses.StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/reset", include_in_schema=False)
    async def gradio_reset():
        return {"ok": True}

    @app.get("/heartbeat/{session_hash}", include_in_schema=False)
    async def gradio_heartbeat(session_hash: str):
        async def beat():
            while True:
                yield "data: " + json.dumps({"msg": "heartbeat"}) + "\n\n"
                await asyncio.sleep(15)

        return fastapi.responses.StreamingResponse(
            beat(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/file={file_path:path}", include_in_schema=False)
    async def gradio_file(file_path: str):
        file_path = file_path.lstrip("/")
        real = os.path.realpath(file_path)
        allowed_roots = [
            os.path.realpath(COVER_OUT_DIR),
            os.path.realpath(tempfile.gettempdir()),
        ]
        if not any(real.startswith(r + os.sep) or real == r for r in allowed_roots):
            raise fastapi.HTTPException(status_code=403, detail="路径不在允许范围内")
        if not os.path.isfile(real):
            raise fastapi.HTTPException(status_code=404, detail="文件不存在")
        return fastapi.responses.FileResponse(real)
