# 代码修改总结

## 修改内容

已将 `infer.py` 改造为可直接在代码中调用的API形式，无需通过命令行参数。

## 新增的主要函数

### 1. `create_detector()` 
创建检测器实例，可重复使用

### 2. `predict_images()`
执行图像检测，返回结果字典

### 3. `predict_images_slice()`
对大图进行切片检测

## 核心特点

✅ **无需命令行** - 直接在Python代码中调用  
✅ **返回结果** - 返回包含检测框的字典，可直接处理  
✅ **可重复使用** - 创建一次detector，多次调用  
✅ **灵活配置** - 所有参数可通过函数参数设置  
✅ **向后兼容** - 原有的命令行方式仍然可用  

## 使用方法

### 最简单的例子

```python
import paddle
from infer import create_detector, predict_images

paddle.enable_static()

# 创建检测器
detector = create_detector(
    model_dir='path/to/model',
    device='CPU',
    threshold=0.5
)

# 预测图像
results = predict_images(
    detector=detector,
    image_file='test.jpg',
    visual=True,
    save_results=True
)

# 使用结果
print(f"检测到 {len(results['boxes'])} 个目标")
for box in results['boxes']:
    class_id, score, x_min, y_min, x_max, y_max = box
    print(f"类别: {int(class_id)}, 置信度: {score:.3f}")
```

## 返回结果格式

```python
results = {
    'boxes': numpy.ndarray,      # shape: [N, 6]
                                 # 每行: [class_id, score, x_min, y_min, x_max, y_max]
    'boxes_num': numpy.ndarray,  # shape: [M]
                                 # M张图片，每张检测到的目标数量
    'masks': numpy.ndarray,      # (可选) 实例分割mask
}
```

## 新增文件说明

1. **API_USAGE.md** - 完整的API使用文档，包含所有参数说明和使用场景
2. **quick_start.py** - 3个最简单的快速开始示例
3. **infer_api_example.py** - 7个详细的使用示例，涵盖各种场景
4. **practical_example.py** - 实际应用示例，展示如何集成到工作流中

## 典型应用场景

### 场景1: 单张图片检测
```python
detector = create_detector(model_dir='model', device='CPU')
results = predict_images(detector, image_file='test.jpg')
```

### 场景2: 批量检测目录
```python
detector = create_detector(model_dir='model', device='GPU', batch_size=4)
results = predict_images(detector, image_dir='images/')
```

### 场景3: 指定图片列表
```python
detector = create_detector(model_dir='model', device='CPU')
results = predict_images(detector, image_list=['a.jpg', 'b.jpg', 'c.jpg'])
```

### 场景4: 只获取结果不保存
```python
results = predict_images(
    detector, 
    image_file='test.jpg',
    visual=False,      # 不保存可视化
    save_results=False # 不保存JSON
)
# 直接在代码中处理numpy数组
```

### 场景5: 大图切片检测
```python
results = predict_images_slice(
    detector,
    image_file='large.jpg',
    slice_size=[640, 640],
    overlap_ratio=[0.25, 0.25]
)
```

## 优势对比

### 原来（命令行方式）
```bash
python infer.py --model_dir=model --image_file=test.jpg --device=GPU
```
- 需要解析命令行参数
- 结果只能保存到文件
- 难以集成到其他Python程序
- 无法复用检测器

### 现在（API调用方式）
```python
detector = create_detector(model_dir='model', device='GPU')
results = predict_images(detector, image_file='test.jpg')
```
- 直接函数调用，参数清晰
- 返回结果字典，可直接处理
- 轻松集成到任何Python程序
- 可复用检测器，提高效率

## 兼容性

✅ 完全向后兼容，原有的命令行方式仍然可用  
✅ `if __name__ == '__main__'` 中的原始逻辑保持不变  
✅ 所有原有功能正常工作  

## 快速测试

1. 查看快速开始示例:
   ```bash
   # 编辑 quick_start.py，修改模型路径和图片路径
   python quick_start.py
   ```

2. 查看完整API文档:
   ```bash
   # 打开 API_USAGE.md 查看详细文档
   ```

3. 查看实际应用示例:
   ```bash
   # 查看 practical_example.py 了解如何集成到工作流
   ```

## 常见问题

**Q: 原来的命令行方式还能用吗？**  
A: 能！完全兼容，`python infer.py --model_dir=...` 仍然可以使用。

**Q: 如何只获取结果不保存文件？**  
A: 设置 `visual=False` 和 `save_results=False`。

**Q: 可以复用检测器吗？**  
A: 可以！创建一次detector后可以多次调用 `predict_images()`。

**Q: 支持批量处理吗？**  
A: 支持！设置 `batch_size` 参数即可批量处理。

## 下一步

1. 根据需要修改示例文件中的模型路径和图片路径
2. 运行 `quick_start.py` 测试基本功能
3. 查看 `API_USAGE.md` 了解更多功能
4. 参考 `practical_example.py` 集成到你的项目中

祝使用愉快！🎉
