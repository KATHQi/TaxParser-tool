"""
Gunicorn 配置文件
用于生产环境部署 Tax Document Parser API
"""
import sys
import os
import multiprocessing

# 添加项目路径到 sys.path，确保能导入 app 模块
# 获取配置文件所在目录（即项目根目录）
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# 绑定地址和端口
bind = "0.0.0.0:8001"

# Worker 进程数（建议 CPU 核心数 * 2 + 1）
workers = 8

# Worker 类型（gthread 支持多线程，适合 I/O 密集型任务）
worker_class = "gthread"

# 每个 worker 的线程数
threads = 2

# 超时时间（秒）- 设置较长避免大文件处理超时
timeout = 60

# 保持活动连接时间
keepalive = 2
# 回收 worker，避免长期被半开连接占满资源
max_requests = 500
max_requests_jitter = 50

# 限制请求头行/字段大小，减少 slowloris 类型探测影响
limit_request_line = 4094
limit_request_fields = 50
limit_request_field_size = 8190


# 日志配置
accesslog = "-"  # 输出到 stdout
errorlog = "-"   # 输出到 stderr
loglevel = "info"

# 进程名称
proc_name = "tax_parser_api"

# SSL 配置（如果需要 HTTPS）
# 检查证书文件是否存在
cert_file_path = os.path.join(project_dir, 'cert.pem')
key_file_path = os.path.join(project_dir, 'key.pem')

if os.path.exists(cert_file_path) and os.path.exists(key_file_path):
    certfile = cert_file_path
    keyfile = key_file_path
    print(f"[gunicorn] 启用 HTTPS: {certfile}, {keyfile}")
else:
    certfile = None
    keyfile = None
    print(f"[gunicorn] 使用 HTTP 模式（未找到证书文件）")

# 预加载应用（减少内存占用，但需要注意线程安全）
preload_app = False  # 设为 False 避免模型加载冲突

# Worker 启动后的钩子：延迟初始化模型
def post_fork(server, worker):
    """Worker 进程 fork 后执行，用于初始化模型"""
    try:
        worker.log.info("post_fork: worker 已 fork，开始在 worker 进程中初始化 parser（若尚未）")
        # 导入 app 模块并调用 get_parser 触发延迟初始化
        # 确保在 worker 进程中初始化模型，避免主进程预加载导致的多进程共享问题
        from app import get_parser
        parser = get_parser()
        worker.log.info("post_fork: parser 初始化成功")
    except Exception as e:
        worker.log.error(f"post_fork: 初始化 parser 失败: {e}")
        # 不抛出异常，让 worker 继续启动，后续请求时会再次尝试初始化

# 应用启动前的钩子
def on_starting(server):
    """Gunicorn 主进程启动时执行"""
    server.log.info("="*60)
    server.log.info("Tax Document Parser API - Gunicorn 启动")
    server.log.info(f"项目目录: {project_dir}")
    server.log.info(f"Workers: {workers}, Threads/Worker: {threads}")
    server.log.info(f"绑定地址: {bind}")
    server.log.info(f"超时时间: {timeout}s")
    if certfile and keyfile:
        server.log.info("协议: HTTPS")
    else:
        server.log.info("协议: HTTP")
    server.log.info("="*60)

# Worker 退出钩子
def worker_exit(server, worker):
    """Worker 进程退出时执行"""
    worker.log.info(f"Worker {worker.pid} 正在退出")
