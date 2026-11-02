"""轮询 RVC 服务是否就绪，就绪后自动打开接口文档页面。

由 启动RVC翻唱服务.bat 以最小化窗口后台调用，不需要任何第三方依赖。
"""
import socket
import sys
import time
import webbrowser

HOST, PORT, TIMEOUT = "127.0.0.1", 8000, 180


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(1.5)
        return s.connect_ex((HOST, PORT)) == 0


def main() -> int:
    start = time.time()
    while time.time() - start < TIMEOUT:
        if port_open():
            time.sleep(1)  # 端口监听后 uvicorn 还需一瞬完成路由注册
            webbrowser.open(f"http://{HOST}:{PORT}/docs")
            print(f"服务已就绪，用时 {time.time() - start:.0f}s，已打开接口文档。")
            return 0
        time.sleep(2)
    print(f"等待 {TIMEOUT}s 后服务仍未就绪，请查看启动窗口的报错。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
