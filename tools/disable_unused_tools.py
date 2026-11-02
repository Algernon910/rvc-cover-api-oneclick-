"""停用 matsuko_cover 插件中用不到的 LLM 函数工具（可在面板随时恢复）。"""
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

HOME = os.path.expanduser("~")  # AstrBot 数据根默认为 ~/.astrbot
sys.path.insert(0, os.path.join(HOME, ".astrbot", "data", "site-packages"))
import jwt

CFG = os.path.join(HOME, ".astrbot", "data", "cmd_config.json")
dashboard = json.load(open(CFG, encoding="utf-8-sig"))["dashboard"]
token = jwt.encode(
    {"username": dashboard.get("username", "astrbot"),
     "exp": datetime.now(timezone.utc) + timedelta(days=7)},
    dashboard["jwt_secret"], algorithm="HS256",
)
HOST = "http://127.0.0.1:6185"
HDR = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def call(method, path, body=None):
    req = urllib.request.Request(
        HOST + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers=HDR,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


code, data = call("GET", "/api/v1/tools")
tools = (data.get("data") or data) if isinstance(data, dict) else data
print("工具总数:", len(tools))

if tools and isinstance(tools[0], dict):
    print("字段:", list(tools[0].keys()))

KEEP = {
    "smart_cover", "rvc_cover", "cover_local_audio",
    "search_music", "get_available_models", "get_task_status",
}
MINE = {
    "smart_cover", "cover_absolute_path_audio", "cover_local_audio", "batch_cover",
    "rvc_cover", "svc_cover", "svcvc_cover", "confirm_selection", "save_preference",
    "search_music", "get_available_models", "get_recommendation", "get_task_status",
    "cancel_cover_task", "clear_my_history", "view_my_stats",
}

targets = []
for t in tools:
    if not isinstance(t, dict):
        continue
    name = t.get("name") or t.get("tool_name") or t.get("id") or ""
    base = str(name).split(":")[-1].split(".")[-1]
    if base in MINE and base not in KEEP:
        targets.append((name, base))

print("\n准备停用的工具:")
for name, base in targets:
    print("  -", base, "(id:", name, ")")

if not targets:
    print("没有匹配到目标工具，请检查名称。")
    raise SystemExit(0)

print()
for name, base in targets:
    code, res = call(
        "PATCH",
        "/api/v1/tools/%s/enabled" % urllib.parse.quote(str(name), safe=""),
        {"enabled": False},
    )
    print(f"{base:28s} -> {code} {str(res)[:90]}")
