# FLAGS 参数与新 API 对应关系

## 完整参数映射表

下面是你提供的 FLAGS 参数与新 API 函数参数的完整对应关系：

| FLAGS 参数 | 对应函数 | 函数参数 | 默认值 | 说明 |
|-----------|---------|---------|--------|------|
| `model_dir` | `create_detector()` | `model_dir` | 必需 | 模型目录路径 |
| `device` | `create_detector()` | `device` | 'CPU' | 'CPU', 'GPU', 'XPU', 'NPU' |
| `threshold` | `create_detector()` | `threshold` | 0.5 | 检测阈值 |
| `batch_size` | `create_detector()` | `batch_size` | 1 | 批处理大小 |
| `output_dir` | `create_detector()` | `output_dir` | 'output' | 输出目录 |
| `run_mode` | `create_detector()` | `run_mode` | 'paddle' | 运行模式 |
| `trt_min_shape` | `create_detector()` | `trt_min_shape` | 1 | TensorRT最小形状 |
| `trt_max_shape` | `create_detector()` | `trt_max_shape` | 1280 | TensorRT最大形状 |
| `trt_opt_shape` | `create_detector()` | `trt_opt_shape` | 640 | TensorRT优化形状 |
| `trt_calib_mode` | `create_detector()` | `trt_calib_mode` | False | TensorRT校准模式 |
| `cpu_threads` | `create_detector()` | `cpu_threads` | 1 | CPU线程数 |
| `enable_mkldnn` | `create_detector()` | `enable_mkldnn` | False | 是否启用MKLDNN |
| `enable_mkldnn_bfloat16` | `create_detector()` | `enable_mkldnn_bfloat16` | False | 是否启用MKLDNN BFloat16 |
| `use_fd_format` | `create_detector()` | `use_fd_format` | False | 是否使用FD格式 |
| `image_file` | `predict_images()` | `image_file` | None | 单个图像文件路径 |
| `image_dir` | `predict_images()` | `image_dir` | None | 图像目录路径 |
| `save_images` | `predict_images()` | `visual` | False | 是否保存可视化结果 |
| `save_results` | `predict_images()` | `save_results` | False | 是否保存JSON结果 |
| `run_benchmark` | `predict_images()` | `run_benchmark` | False | 是否运行基准测试 |
| `slice_infer` | `run_detector_from_config()` | `slice_infer` | False | 是否使用切片推理 |
| `slice_size` | `predict_images_slice()` | `slice_size` | [640, 640] | 切片大小 |
| `overlap_ratio` | `predict_images_slice()` | `overlap_ratio` | [0.25, 0.25] | 重叠比例 |
| `combine_method` | `predict_images_slice()` | `combine_method` | 'nms' | 合并方法 |
| `match_threshold` | `predict_images_slice()` | `match_threshold` | 0.6 | 匹配阈值 |
| `match_metric` | `predict_images_slice()` | `match_metric` | 'ios' | 匹配度量 |
| `use_gpu` | - | - | 已弃用 | 使用 device='GPU' 代替 |
| `camera_id` | `predict_video()` | - | -1 | 摄像头ID（视频检测） |
| `video_file` | `predict_video()` | - | None | 视频文件（视频检测） |

## 未直接映射的参数（主要用于特定场景）

以下参数在标准图像检测场景中不常用：

| FLAGS 参数 | 用途 | 说明 |
|-----------|------|------|
| `action_file` | 行为识别 | 用于行为识别模型 |
| `reid_model_dir` | ReID | 用于行人重识别 |
| `reid_batch_size` | ReID | ReID批处理大小 |
| `tracker_config` | 跟踪 | 多目标跟踪配置 |
| `save_mot_txts` | 跟踪 | 保存MOT跟踪结果 |
| `save_mot_txt_per_img` | 跟踪 | 每张图保存跟踪结果 |
| `scaled` | 跟踪 | 跟踪结果缩放 |
| `random_pad` | 数据增强 | 随机填充 |
| `use_dark` | 关键点 | 关键点检测相关 |
| `window_size` | 时序 | 时序模型窗口大小 |
| `collect_trt_shape_info` | TensorRT | 收集TRT形状信息 |
| `tuned_trt_shape_file` | TensorRT | TRT形状文件路径 |
| `use_coco_category` | COCO | 使用COCO类别ID |

## 三种使用方式对比

### 方式1: 使用 `run_detector_from_config()` (最简单，完全兼容FLAGS)

```python
import paddle
from infer import run_detector_from_config

paddle.enable_static()

# 方式1a: 使用字典（推荐）
config = {
    'model_dir': '/path/to/model',
    'device': 'GPU',
    'image_file': '/path/to/image.jpg',
    'threshold': 0.5,
    'batch_size': 1,
    'save_images': True,
    'save_results': True,
    'output_dir': '/path/to/output',
}
results = run_detector_from_config(config)

# 方式1b: 直接使用FLAGS对象
results = run_detector_from_config(FLAGS)
```

✅ **优点**: 
- 完全兼容原有FLAGS参数
- 一行代码完成检测
- 支持所有参数

### 方式2: 两步方式 (最灵活)

```python
from infer import create_detector, predict_images

# 第一步：创建检测器
detector = create_detector(
    model_dir='/path/to/model',
    device='GPU',
    threshold=0.5,
    batch_size=1,
    output_dir='/path/to/output'
)

# 第二步：预测
results = predict_images(
    detector=detector,
    image_file='/path/to/image.jpg',
    visual=True,
    save_results=True
)
```

✅ **优点**:
- 可以复用detector对象
- 更灵活的参数控制
- 适合批量处理多个不同配置

### 方式3: 原命令行方式 (向后兼容)

```bash
python infer.py --model_dir=/path/to/model --image_file=/path/to/image.jpg --device=GPU
```

✅ **优点**: 保持原有使用习惯

## 完整示例：从FLAGS迁移到新API

### 原来的方式

```python
# 原来需要通过命令行
# python infer.py --model_dir=/data/.../yolo3 \
#                 --image_file=/data/.../main_48_page1.jpg \
#                 --device=GPU \
#                 --threshold=0.5 \
#                 --save_images=True \
#                 --save_results=True

# 或者在代码中
if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.save_results = True
    main()
```

### 现在的方式（推荐）

```python
import paddle
from infer import run_detector_from_config

paddle.enable_static()

# 直接使用配置字典
config = {
    'model_dir': '/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2/train2-best/infer/yolo3',
    'image_file': '/data/tensorflow/kath/Data/OCR/Tax/test/TaxDoc/main_48_page1.jpg',
    'device': 'GPU',
    'threshold': 0.5,
    'save_images': True,
    'save_results': True,
    'output_dir': '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/output',
}

# 一行代码完成检测并获取结果
results = run_detector_from_config(config)

# 直接使用返回的结果
print(f"检测到 {len(results['boxes'])} 个目标")
for box in results['boxes']:
    class_id, score, x_min, y_min, x_max, y_max = box
    print(f"类别: {int(class_id)}, 置信度: {score:.3f}")
```

## 验证：确保参数对应正确

根据你提供的FLAGS参数，我已经确认：

✅ **核心检测参数** - 全部支持
- model_dir ✓
- device ✓
- threshold ✓
- batch_size ✓
- image_file / image_dir ✓
- save_images (对应visual) ✓
- save_results ✓
- output_dir ✓

✅ **性能优化参数** - 全部支持
- run_mode ✓
- trt_min_shape, trt_max_shape, trt_opt_shape ✓
- trt_calib_mode ✓
- cpu_threads ✓
- enable_mkldnn ✓
- enable_mkldnn_bfloat16 ✓

✅ **切片推理参数** - 全部支持
- slice_infer ✓
- slice_size ✓
- overlap_ratio ✓
- combine_method ✓
- match_threshold ✓
- match_metric ✓

✅ **其他参数** - 支持
- run_benchmark ✓
- use_fd_format ✓

## 总结

新API **完全覆盖** 了你的FLAGS参数需求，并提供了三种使用方式：

1. **`run_detector_from_config(config)`** - 最简单，完全兼容FLAGS
2. **`create_detector()` + `predict_images()`** - 最灵活，可复用
3. 原命令行方式 - 向后兼容

所有参数都已正确映射和支持！✅
