# Tax Document Parser - Flask API Service

一个专业的税务文档在线解析服务，基于 PaddleOCR 和深度学习的版面分析技术。

## 功能特性

- 🚀 **实时处理**: 支持单个或批量图像处理
- 📊 **智能布局分析**: 自动识别表格、标题等区域
- 🔤 **高精度 OCR**: 基于 PaddleOCR 的中文识别
- 📋 **表格提取**: 自动提取表格结构和数据
- 🎨 **可视化展示**: 美观的前端界面
- 📈 **性能优化**: 支持 GPU 加速

## 快速开始

### 1. 环境要求

- Python 3.7+
- CUDA 支持（可选但推荐）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python app.py
```

服务将运行在 `http://localhost:8003`

### 4. 访问 Web 界面

打开浏览器，访问:
```
http://localhost:8003
```

## API 文档

### 1. 解析文档

**端点**: `POST /api/parse`

**请求体**:
```json
{
    "directory_path": "/path/to/images/folder",
    "recursive": false
}
```

**响应示例**:
```json
{
    "status": "success",
    "data": {
        "total_images": 5,
        "successful": 5,
        "failed": 0,
        "images": [
            {
                "status": "success",
                "image_path": "/path/to/image.jpg",
                "grouped_data": {
                    "table": {"count": 2, "items": []},
                    "title": {"count": 1, "items": []}
                },
                "tables": {
                    "table": {
                        "html": "<table>...</table>",
                        "rows": 5,
                        "cols": 3
                    }
                }
            }
        ]
    }
}
```

### 2. 健康检查

**端点**: `GET /api/health`

**响应**:
```json
{
    "status": "ok",
    "service": "Tax Document Parser API",
    "version": "1.0.0"
}
```

### 3. 服务状态

**端点**: `GET /api/status`

**响应**:
```json
{
    "status": "running",
    "service": "Tax Document Parser API",
    "port": 8003
}
```

## 使用示例

### Python 调用示例

```python
import requests
import json

# 服务地址
API_URL = "http://localhost:8003/api/parse"

# 请求数据
payload = {
    "directory_path": "F:/images/tax_docs",
    "recursive": False
}

# 发送请求
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### cURL 调用示例

```bash
curl -X POST http://localhost:8003/api/parse \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "F:/images/tax_docs", "recursive": false}'
```

## 配置说明

### Flask 配置

在 `app.py` 中可以调整以下参数:

- `host`: 服务监听地址（默认: 0.0.0.0）
- `port`: 服务端口（默认: 8003）
- `debug`: 调试模式（默认: False）
- `threaded`: 线程模式（默认: True）

### 模型配置

原始算法模块保存在 `codeData` 文件夹中，主要模型配置包括:

- OCR 模型: PaddleOCR
- 布局分析模型: PaddleDetection
- 表格识别模型: SLANet

## 前端功能

### 输入界面

- 📁 文件夹路径输入
- ✓ 递归处理选项
- ⚡ 快速开始按钮

### 结果展示

- 📊 统计卡片：显示处理统计
- 📋 布局信息：显示各布局区域的文本数量
- 📈 表格数据：HTML 格式表格展示
- 🔍 详细信息：逐个图像的处理结果

## 目录结构

```
TaxParser-API/
├── app.py                 # Flask 主应用
├── requirements.txt       # 依赖列表
├── templates/
│   └── index.html        # 前端页面
├── static/               # 静态文件（CSS/JS）
├── codeData/             # 原始算法模块（软链接或复制）
└── README.md             # 本文件
```

## 故障排除

### 1. 模型加载失败

**问题**: `ImportError: cannot import name 'PaddleOCR'`

**解决方案**:
```bash
pip install paddleocr paddlepaddle
```

### 2. CUDA 相关错误

**问题**: `RuntimeError: cuda runtime error`

**解决方案**: 
- 确保 CUDA 版本与 PyTorch/PaddlePaddle 兼容
- 或禁用 GPU: 在 `app.py` 中设置 `use_gpu=False`

### 3. 路径错误

**问题**: `No such file or directory`

**解决方案**:
- 确保输入的文件夹路径存在
- 检查路径分隔符（Windows 用 `\` 或 `/`）

### 4. 内存不足

**问题**: `MemoryError` 或进程被杀死

**解决方案**:
- 减少批处理大小
- 使用 CPU 代替 GPU
- 分批处理文件

## 性能优化

### 1. GPU 加速

确保系统安装了支持 GPU 的 PyTorch/PaddlePaddle

### 2. 模型缓存

应用会在首次启动时加载模型，之后会被缓存

### 3. 并发处理

Flask 服务支持多线程，可同时处理多个请求

## 生产部署

### 使用 Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8003 app:app
```

### 使用 Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8003
CMD ["python", "app.py"]
```

### 使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://127.0.0.1:8003;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 许可证

MIT License

## 技术栈

- **后端**: Flask
- **OCR**: PaddleOCR
- **布局分析**: PaddleDetection
- **表格识别**: SLANet
- **前端**: HTML5 + CSS3 + Vanilla JavaScript

## 支持

如有问题，请查看日志或提交 Issue。

---

**版本**: 1.0.0  
**最后更新**: 2025-11-13
