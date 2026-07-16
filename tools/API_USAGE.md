# 目标检测推理 API 使用说明

## 概述

现在你可以通过代码直接调用 `infer.py` 进行目标检测，无需使用命令行参数。

## 主要函数

### 1. `create_detector()` - 创建检测器

创建一个检测器实例用于后续预测。

```python
from infer import create_detector
import paddle

paddle.enable_static()

detector = create_detector(
    model_dir='path/to/model',  # 模型目录路径（必需）
    device='CPU',               # 设备: 'CPU', 'GPU', 'XPU', 'NPU'
    run_mode='paddle',          # 运行模式: 'paddle', 'trt_fp32', 'trt_fp16', 'trt_int8'
    batch_size=1,               # 批处理大小
    threshold=0.5,              # 检测阈值
    output_dir='output',        # 输出目录
    use_fd_format=False         # 是否使用FD格式
)
```

### 2. `predict_images()` - 图像预测

对图像进行目标检测预测。

```python
from infer import predict_images

results = predict_images(
    detector=detector,           # 检测器实例（必需）
    image_list=['a.jpg', 'b.jpg'],  # 图像路径列表（三选一）
    image_file='test.jpg',       # 单个图像文件（三选一）
    image_dir='images/',         # 图像目录（三选一）
    run_benchmark=False,         # 是否运行基准测试
    repeats=1,                   # 重复预测次数
    visual=True,                 # 是否保存可视化结果
    save_results=True            # 是否保存JSON结果
)
```

**返回结果格式:**
```python
{
    'boxes': numpy.ndarray,      # shape: [N, 6], 每行: [class_id, score, x_min, y_min, x_max, y_max]
    'boxes_num': numpy.ndarray,  # shape: [M], 每张图片检测到的目标数量
    'masks': numpy.ndarray,      # (可选) 如果模型支持实例分割
}
```

### 3. `predict_images_slice()` - 切片预测

用于大分辨率图像的切片预测。

```python
from infer import predict_images_slice

results = predict_images_slice(
    detector=detector,
    image_file='large_image.jpg',
    slice_size=[640, 640],       # 切片大小 [height, width]
    overlap_ratio=[0.25, 0.25],  # 重叠比例 [height_ratio, width_ratio]
    combine_method='nms',        # 合并方法: 'nms' 或 'concat'
    match_threshold=0.6,         # NMS阈值
    match_metric='ios',          # 匹配度量: 'ios' 或 'iou'
    visual=True,
    save_results=True
)
```

## 快速开始

### 最简单的示例

```python
import paddle
from infer import create_detector, predict_images

# 初始化
paddle.enable_static()

# 创建检测器
detector = create_detector(
    model_dir='path/to/your/model',
    device='CPU',
    threshold=0.5
)

# 预测单张图片
results = predict_images(
    detector=detector,
    image_file='test.jpg',
    visual=True,
    save_results=True
)

# 查看结果
print(f"检测到 {len(results['boxes'])} 个目标")
for box in results['boxes']:
    class_id, score, x_min, y_min, x_max, y_max = box
    print(f"类别: {int(class_id)}, 置信度: {score:.3f}")
```

## 使用场景示例

### 场景1: 批量处理目录中的图片

```python
detector = create_detector(model_dir='model_path', device='GPU', batch_size=4)

results = predict_images(
    detector=detector,
    image_dir='test_images/',
    visual=True,
    save_results=True
)

print(f"处理了 {len(results['boxes_num'])} 张图片")
```

### 场景2: 只获取结果不保存文件

```python
detector = create_detector(model_dir='model_path', device='CPU')

results = predict_images(
    detector=detector,
    image_file='test.jpg',
    visual=False,      # 不保存可视化图片
    save_results=False # 不保存JSON文件
)

# 直接在内存中处理结果
boxes = results['boxes']
high_conf_boxes = boxes[boxes[:, 1] > 0.8]  # 筛选高置信度目标
```

### 场景3: 预测指定的图片列表

```python
detector = create_detector(model_dir='model_path', device='CPU')

my_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']

results = predict_images(
    detector=detector,
    image_list=my_images,
    visual=True,
    save_results=False
)

# 逐个处理每张图片的结果
start_idx = 0
for i, boxes_num in enumerate(results['boxes_num']):
    print(f"图片 {my_images[i]} 检测到 {boxes_num} 个目标")
    img_boxes = results['boxes'][start_idx:start_idx + boxes_num]
    start_idx += boxes_num
```

### 场景4: 大图切片预测

```python
detector = create_detector(model_dir='model_path', device='GPU')

results = predict_images_slice(
    detector=detector,
    image_file='high_resolution_image.jpg',
    slice_size=[640, 640],
    overlap_ratio=[0.25, 0.25],
    combine_method='nms',
    visual=True
)
```

### 场景5: 使用TensorRT加速

```python
detector = create_detector(
    model_dir='model_path',
    device='GPU',
    run_mode='trt_fp16',  # 使用TensorRT FP16
    batch_size=8,
    trt_min_shape=320,
    trt_max_shape=1280,
    trt_opt_shape=640
)

results = predict_images(
    detector=detector,
    image_dir='images/',
    visual=True
)
```

## 结果解析

### 访问检测框

```python
results = predict_images(detector, image_file='test.jpg')

# 获取所有检测框
boxes = results['boxes']  # shape: [N, 6]

for box in boxes:
    class_id = int(box[0])      # 类别ID
    confidence = box[1]         # 置信度
    x_min = box[2]              # 左上角x坐标
    y_min = box[3]              # 左上角y坐标
    x_max = box[4]              # 右下角x坐标
    y_max = box[5]              # 右下角y坐标
    
    # 计算宽高
    width = x_max - x_min
    height = y_max - y_min
```

### 处理多张图片的结果

```python
results = predict_images(detector, image_list=['a.jpg', 'b.jpg', 'c.jpg'])

start_idx = 0
for img_idx, boxes_num in enumerate(results['boxes_num']):
    # 获取当前图片的所有检测框
    end_idx = start_idx + boxes_num
    img_boxes = results['boxes'][start_idx:end_idx]
    
    print(f"图片 {img_idx} 的检测结果:")
    for box in img_boxes:
        print(f"  类别: {int(box[0])}, 置信度: {box[1]:.3f}")
    
    start_idx = end_idx
```

### 筛选结果

```python
import numpy as np

results = predict_images(detector, image_file='test.jpg')
boxes = results['boxes']

# 筛选特定类别
class_id = 0
class_boxes = boxes[boxes[:, 0] == class_id]

# 筛选高置信度
high_conf_boxes = boxes[boxes[:, 1] > 0.8]

# 筛选特定区域
# 例如: 只要图像左半部分的检测框
left_half_boxes = boxes[boxes[:, 4] < image_width / 2]
```

## 与原命令行方式对比

### 原来的方式（命令行）
```bash
python infer.py --model_dir=model --image_file=test.jpg --device=GPU --threshold=0.5
```

### 现在的方式（代码调用）
```python
import paddle
from infer import create_detector, predict_images

paddle.enable_static()

detector = create_detector(
    model_dir='model',
    device='GPU',
    threshold=0.5
)

results = predict_images(
    detector=detector,
    image_file='test.jpg',
    visual=True,
    save_results=True
)
```

## 常见问题

### Q1: 如何只获取检测结果而不保存文件？
```python
results = predict_images(
    detector=detector,
    image_file='test.jpg',
    visual=False,      # 不保存可视化
    save_results=False # 不保存JSON
)
```

### Q2: 如何批量处理大量图片？
```python
detector = create_detector(model_dir='model', device='GPU', batch_size=8)
results = predict_images(detector, image_dir='large_dataset/')
```

### Q3: 检测结果保存在哪里？
- 可视化图片: `output_dir` 参数指定的目录（默认 'output/'）
- JSON结果: `output_dir/bbox.json`

### Q4: 如何复用检测器？
```python
# 创建一次检测器
detector = create_detector(model_dir='model', device='GPU')

# 多次使用
results1 = predict_images(detector, image_file='img1.jpg')
results2 = predict_images(detector, image_file='img2.jpg')
results3 = predict_images(detector, image_dir='test_images/')
```

## 参考文件

- `infer_api_example.py` - 详细的使用示例
- `quick_start.py` - 快速开始示例
- `infer.py` - 主要API实现

## 注意事项

1. 使用前必须调用 `paddle.enable_static()`
2. `model_dir` 必须包含 `model.pdmodel`、`model.pdiparams` 和 `infer_cfg.yml` 文件
3. 使用 GPU 时确保已正确安装 PaddlePaddle GPU 版本
4. `image_list`、`image_file` 和 `image_dir` 三个参数只需提供一个
5. 设置 `visual=True` 会在 `output_dir` 中保存可视化结果图片
6. 设置 `save_results=True` 会在 `output_dir` 中保存 JSON 格式的检测结果
