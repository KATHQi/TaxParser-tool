"""
使用类似FLAGS配置的示例
演示如何从配置字典或FLAGS对象运行检测器
"""

import paddle
from infer import create_detector, predict_images, run_detector_from_config

# 初始化
paddle.enable_static()


def example_with_dict_config():
    """使用字典配置（推荐方式）"""
    print("=" * 60)
    print("示例1: 使用字典配置")
    print("=" * 60)
    
    # 创建类似FLAGS的配置字典
    config = {
        'model_dir': '/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2/train2-best/infer/yolo3',
        'device': 'GPU',
        'image_file': '/data/tensorflow/kath/Data/OCR/Tax/test/TaxDoc/main_48_page1.jpg',
        'threshold': 0.5,
        'batch_size': 1,
        'save_images': True,
        'save_results': True,
        'output_dir': '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/output',
        'run_mode': 'paddle',
        'slice_infer': False,
        'run_benchmark': False,
        'use_fd_format': False
    }
    
    # 一行代码完成检测
    results = run_detector_from_config(config)
    
    print(f"\n检测完成！检测到 {len(results['boxes'])} 个目标")
    
    return results


def example_with_flags_object(FLAGS):
    """使用FLAGS对象（兼容原有代码）"""
    print("=" * 60)
    print("示例2: 使用FLAGS对象")
    print("=" * 60)
    
    # 直接使用FLAGS对象
    results = run_detector_from_config(FLAGS)
    
    print(f"\n检测完成！检测到 {len(results['boxes'])} 个目标")
    
    return results


def example_manual_two_steps():
    """两步方式：先创建检测器，再预测（更灵活）"""
    print("=" * 60)
    print("示例3: 两步方式（更灵活）")
    print("=" * 60)
    
    # 第一步：创建检测器
    detector = create_detector(
        model_dir='/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2/train2-best/infer/yolo3',
        device='GPU',
        threshold=0.5,
        batch_size=1,
        output_dir='/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/output',
        run_mode='paddle'
    )
    
    # 第二步：预测图像
    results = predict_images(
        detector=detector,
        image_file='/data/tensorflow/kath/Data/OCR/Tax/test/TaxDoc/main_48_page1.jpg',
        visual=True,
        save_results=True
    )
    
    print(f"\n检测完成！检测到 {len(results['boxes'])} 个目标")
    
    # 优势：可以复用detector处理多张图片
    # results2 = predict_images(detector, image_file='another_image.jpg')
    
    return results


def example_batch_processing():
    """批量处理多张图片"""
    print("=" * 60)
    print("示例4: 批量处理")
    print("=" * 60)
    
    config = {
        'model_dir': '/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2/train2-best/infer/yolo3',
        'device': 'GPU',
        'image_dir': '/data/tensorflow/kath/Data/OCR/Tax/test/TaxDoc/',  # 整个目录
        'threshold': 0.5,
        'batch_size': 4,  # 批量处理
        'save_images': True,
        'save_results': True,
        'output_dir': '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/output'
    }
    
    results = run_detector_from_config(config)
    
    print(f"\n批量检测完成！")
    print(f"处理了 {len(results['boxes_num'])} 张图片")
    print(f"总共检测到 {len(results['boxes'])} 个目标")
    
    return results


def example_with_slice_infer():
    """使用切片推理（适用于大图）"""
    print("=" * 60)
    print("示例5: 切片推理")
    print("=" * 60)
    
    config = {
        'model_dir': '/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2/train2-best/infer/yolo3',
        'device': 'GPU',
        'image_file': '/data/tensorflow/kath/Data/OCR/Tax/test/TaxDoc/main_48_page1.jpg',
        'threshold': 0.5,
        'slice_infer': True,  # 开启切片推理
        'slice_size': [640, 640],
        'overlap_ratio': [0.25, 0.25],
        'combine_method': 'nms',
        'match_threshold': 0.6,
        'match_metric': 'ios',
        'save_images': True,
        'save_results': True,
        'output_dir': '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/output'
    }
    
    results = run_detector_from_config(config)
    
    print(f"\n切片检测完成！检测到 {len(results['boxes'])} 个目标")
    
    return results


def example_process_results(results):
    """处理检测结果"""
    print("\n" + "=" * 60)
    print("处理检测结果")
    print("=" * 60)
    
    boxes = results['boxes']
    boxes_num = results['boxes_num']
    
    print(f"\n总检测数: {len(boxes)}")
    print(f"图片数: {len(boxes_num)}")
    
    # 统计每个类别的数量
    class_counts = {}
    for box in boxes:
        class_id = int(box[0])
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    print("\n各类别检测数量:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  类别 {class_id}: {count}")
    
    # 显示前几个检测结果
    print("\n前5个检测结果:")
    for i, box in enumerate(boxes[:5]):
        class_id, score, x_min, y_min, x_max, y_max = box
        print(f"  {i+1}. 类别={int(class_id)}, 置信度={score:.3f}, "
              f"位置=({x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f})")
    
    # 筛选高置信度结果
    high_conf_boxes = boxes[boxes[:, 1] > 0.8]
    print(f"\n高置信度(>0.8)的目标: {len(high_conf_boxes)}")


def example_compatible_with_original_code():
    """完全兼容原始代码的方式"""
    print("=" * 60)
    print("示例6: 兼容原始代码")
    print("=" * 60)
    print("这个方式完全等同于运行命令行:")
    print("python infer.py --model_dir=... --image_file=... --device=GPU")
    
    # 模拟FLAGS对象
    class MockFlags:
        def __init__(self):
            self.model_dir = '/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2/train2-best/infer/yolo3'
            self.device = 'GPU'
            self.image_file = '/data/tensorflow/kath/Data/OCR/Tax/test/TaxDoc/main_48_page1.jpg'
            self.image_dir = None
            self.threshold = 0.5
            self.batch_size = 1
            self.save_images = True
            self.save_results = True
            self.output_dir = '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/output'
            self.run_mode = 'paddle'
            self.run_benchmark = False
            self.slice_infer = False
            self.slice_size = [640, 640]
            self.overlap_ratio = [0.25, 0.25]
            self.combine_method = 'nms'
            self.match_threshold = 0.6
            self.match_metric = 'ios'
            self.trt_min_shape = 1
            self.trt_max_shape = 1280
            self.trt_opt_shape = 640
            self.trt_calib_mode = False
            self.cpu_threads = 1
            self.enable_mkldnn = False
            self.enable_mkldnn_bfloat16 = False
            self.use_fd_format = False
            self.repeats = 1
    
    flags = MockFlags()
    results = run_detector_from_config(flags)
    
    print(f"\n检测完成！检测到 {len(results['boxes'])} 个目标")
    
    return results


# ============= 主程序 =============

if __name__ == '__main__':
    # 选择要运行的示例
    
    # 推荐方式1: 使用字典配置（最简单）
    results = example_with_dict_config()
    
    # 推荐方式2: 两步方式（最灵活）
    # results = example_manual_two_steps()
    
    # 其他示例
    # results = example_batch_processing()
    # results = example_with_slice_infer()
    # results = example_compatible_with_original_code()
    
    # 处理结果
    example_process_results(results)
    
    print("\n" + "=" * 60)
    print("✅ 检测完成！")
    print("=" * 60)
    print("\n提示:")
    print("1. 结果已保存到 output_dir 指定的目录")
    print("2. 可视化图片: output_dir/*.jpg")
    print("3. JSON结果: output_dir/bbox.json")
    print("4. results 字典可直接在代码中使用")
