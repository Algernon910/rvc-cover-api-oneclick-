"""完全模拟 matsuko 插件的调用方式（关键字参数 + 默认配置值）打 /convert。"""
import os
import sys
import time

HOME = os.path.expanduser("~")  # AstrBot 数据根默认为 ~/.astrbot
sys.path.insert(0, os.path.join(HOME, ".astrbot", "data", "site-packages"))
from gradio_client import Client

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "song_test.wav"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "my_voice.pth"

client = Client("http://127.0.0.1:8000/", verbose=False, analytics_enabled=False)

# matsuko 插件 main.py _predict_cover() 的 legacy_kwargs（默认配置取值）
kwargs = {
    "song_name_src": AUDIO,
    "key_shift": 0,
    "vocal_vol": 0,
    "inst_vol": 0,
    "model_dropdown": MODEL,
    "reverb_intensity": 4,
    "delay_intensity": 0,
    "uvr5_agg": 10,
    "uvr5_tta": False,
    "uvr5_postprocess": False,
    "uvr5_window_size": 512,
    "uvr5_high_end_process": "mirroring",
    "msst_batch_size": 1,
    "msst_num_overlap": 4,
    "msst_normalize": False,
    "vocal_postprocess": False,
    "shift_accompaniment": True,
    "f0_method": "rmvpe",
    "index_rate": "0.75",
    "filter_radius": 3,
}

print(f"提交: 模型={MODEL} 音频={AUDIO}")
t0 = time.time()
job = client.submit(**kwargs, api_name="/convert")
while True:
    st = job.status()
    pd = getattr(st, "progress_data", None) or []
    code = getattr(st, "code", None)
    if pd:
        for item in pd:
            desc = getattr(item, "desc", None) or (item.get("desc") if isinstance(item, dict) else "")
            if desc:
                print("  [进度]", desc)
    if code and getattr(code, "name", getattr(code, "value", code)) == "FINISHED":
        break
    if code and getattr(code, "name", getattr(code, "value", code)) in ("FAILED", "CANCELLED"):
        print("任务失败:", st)
        raise SystemExit(1)
    time.sleep(1)
    if time.time() - t0 > 600:
        print("超时")
        raise SystemExit(1)

result = job.result()
print(f"[ok] 用时 {time.time()-t0:.1f}s")
print("返回类型:", type(result), "内容:", str(result)[:500])

# 检查是否真的出了文件
try:
    p = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(p, dict):
        p = p.get("path") or p.get("name")
    if isinstance(p, str) and os.path.exists(p):
        print(f"[ok] 音频文件存在: {p} ({os.path.getsize(p)/1024:.0f} KB)")
    else:
        print("[warn] 未拿到真实文件:", p)
except Exception as e:
    print("解析结果失败:", e)
print("cache_flag:", result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else None)
client.close()
