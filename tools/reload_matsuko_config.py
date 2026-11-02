"""通过 AstrBot dashboard API 热更新 matsuko 插件配置（改完立即生效，无需重启机器人）。

用法
----
    python reload_matsuko_config.py                        # 只切换音乐源
    python reload_matsuko_config.py voice_a.pth voice_b.pth # 同时写入音色列表
    set RVC_DEFAULT_MODEL=voice_a.pth && python reload_matsuko_config.py

原理：AstrBot 的 dashboard 用 HS256 JWT 鉴权，payload 只需 username + exp，
密钥就在本地 cmd_config.json 的 dashboard.jwt_secret 里 —— 所以不需要面板密码。
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

HOME = os.path.expanduser("~")  # AstrBot 数据根默认为 ~/.astrbot
sys.path.insert(0, os.path.join(HOME, ".astrbot", "data", "site-packages"))
import jwt  # AstrBot 环境自带 PyJWT

CFG = os.path.join(HOME, ".astrbot", "data", "cmd_config.json")
base = json.load(open(CFG, encoding="utf-8-sig"))["dashboard"]
SECRET = base["jwt_secret"]
USERNAME = base.get("username", "astrbot")
HOST = "http://127.0.0.1:6185"

token = jwt.encode(
    {"username": USERNAME, "exp": datetime.now(timezone.utc) + timedelta(days=7)},
    SECRET, algorithm="HS256",
)
HDR = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def call(method, path, body=None):
    req = urllib.request.Request(
        HOST + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers=HDR,
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


PID = "astrbot_plugin_matsuko_cover"

# 1) 读当前配置
code, data = call("GET", f"/api/v1/plugins/config?plugin_id={PID}")
print("GET config:", code)
if code != 200:
    print(str(data)[:500])
    raise SystemExit(1)
conf = data["data"].get("config") or data["data"]
print("当前 rvc_models_keywords:", conf.get("rvc_models_keywords"))

# 2) 写入音色列表（命令行传入的文件名；格式 <模型名>.pth||| ，过滤掉 _e*_s* 中间权重更清爽）
models = [a for a in sys.argv[1:] if a.strip()]
if models:
    conf["rvc_models_keywords"] = [m if "|||" in m else f"{m}|||" for m in models]
    print("将写入:", conf["rvc_models_keywords"])

# 3) 音乐源：旧的第三方代理站已不可用，统一走网易云官方搜索接口
conf["default_api"] = "netease"

# 4) 可选：/本地翻唱 不带参数时使用的默认音色
default_model = os.getenv("RVC_DEFAULT_MODEL", "").strip()
if default_model:
    conf["local_audio_default_model"] = default_model
    print("默认音色 ->", default_model)

code, data = call("PUT", "/api/v1/plugins/config", {"plugin_id": PID, "config": conf})
print("PUT config:", code, str(data)[:300])
