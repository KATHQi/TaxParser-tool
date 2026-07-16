"""
使用infer.py的API示例
演示如何通过代码调用检测功能
"""

import paddle
from infer import create_detector, predict_images, predict_images_slice

# 初始化Paddle
paddle.enable_static()


def example_basic_usage():
    """基础使用示例"""
    print("=" * 50)
    print("基础使用示例")
    print("=" * 50)
    
    # 1. 创建检测器
    detector = create_detector(
        model_dir='your_model_path',  # 替换为你的模型路径
        device='CPU',  # 或 'GPU'
        threshold=0.5,
        output_dir='output'
    )
    
    # 2. 对单张图片进行预测
    results = predict_images(
        detector=detector,
        image_file='test_image.jpg',  # 替换为你的图片路径
        visual=True,  # 是否保存可视化结果
        save_results=True  # 是否保存JSON结果
    )
    
    # 3. 处理结果
    print(f"检测到 {len(results['boxes'])} 个目标")
    print("结果包含的键:", results.keys())
    
    # boxes格式: [class_id, score, x_min, y_min, x_max, y_max]
    for i, box in enumerate(results['boxes']):
        class_id, score, x_min, y_min, x_max, y_max = box
        print(f"目标 {i+1}: 类别={int(class_id)}, 置信度={score:.3f}, "
              f"位置=({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f})")
    
    return results


def example_batch_prediction():
    """批量预测示例"""
    print("\n" + "=" * 50)
    print("批量预测示例")
    print("=" * 50)
    
    # 创建检测器
    detector = create_detector(
        model_dir='your_model_path',
        device='GPU',  # 批量预测建议使用GPU
        batch_size=4,  # 批处理大小
        threshold=0.5
    )
    
    # 对目录中的所有图片进行预测
    results = predict_images(
        detector=detector,
        image_dir='test_images/',  # 替换为你的图片目录
        visual=True,
        save_results=True
    )
    
    print(f"总共检测到 {len(results['boxes'])} 个目标")
    print(f"处理了 {len(results['boxes_num'])} 张图片")
    
    return results


def example_with_image_list():
    """使用图片列表进行预测"""
    print("\n" + "=" * 50)
    print("使用图片列表预测示例")
    print("=" * 50)
    
    # 创建检测器
    detector = create_detector(
        model_dir='your_model_path',
        device='CPU',
        threshold=0.5
    )
    
    # 指定要预测的图片列表
    image_list = [
        'image1.jpg',
        'image2.jpg',
        'image3.jpg'
    ]
    
    # 进行预测
    results = predict_images(
        detector=detector,
        image_list=image_list,
        visual=False,  # 不保存可视化
        save_results=False  # 不保存JSON
    )
    
    # 逐个处理每张图片的结果
    start_idx = 0
    for img_idx, boxes_num in enumerate(results['boxes_num']):
        print(f"\n图片 {image_list[img_idx]}:")
        print(f"  检测到 {boxes_num} 个目标")
        
        # 获取该图片的检测框
        img_boxes = results['boxes'][start_idx:start_idx + boxes_num]
        for box in img_boxes:
            class_id, score = int(box[0]), box[1]
            print(f"    类别={class_id}, 置信度={score:.3f}")
        
        start_idx += boxes_num
    
    return results


def example_slice_prediction():
    """大图切片预测示例（用于高分辨率图像）"""
    print("\n" + "=" * 50)
    print("切片预测示例")
    print("=" * 50)
    
    # 创建检测器
    detector = create_detector(
        model_dir='your_model_path',
        device='GPU',
        threshold=0.5
    )
    
    # 对大图进行切片预测
    results = predict_images_slice(
        detector=detector,
        image_file='large_image.jpg',
        slice_size=[640, 640],  # 切片大小
        overlap_ratio=[0.25, 0.25],  # 重叠比例
        combine_method='nms',  # 合并方法: 'nms' 或 'concat'
        match_threshold=0.6,  # NMS阈值
        visual=True,
        save_results=True
    )
    
    print(f"检测到 {len(results['boxes'])} 个目标")
    
    return results


def example_return_only_results():
    """只获取结果，不保存任何文件"""
    print("\n" + "=" * 50)
    print("只获取结果示例")
    print("=" * 50)
    
    detector = create_detector(
        model_dir='your_model_path',
        device='CPU',
        threshold=0.5,
        output_dir='temp_output'  # 临时输出目录
    )
    
    # 不保存任何可视化或JSON文件，只返回结果
    results = predict_images(
        detector=detector,
        image_file='test.jpg',
        visual=False,  # 不保存可视化图片
        save_results=False  # 不保存JSON文件
    )
    
    # 直接在代码中使用结果
    boxes = results['boxes']
    boxes_num = results['boxes_num'][0]
    
    print(f"返回结果字典的键: {results.keys()}")
    print(f"检测框数量: {boxes_num}")
    print(f"检测框数组形状: {boxes.shape}")
    
    # 可以直接处理numpy数组
    import numpy as np
    high_conf_boxes = boxes[boxes[:, 1] > 0.8]  # 筛选置信度>0.8的框
    print(f"高置信度(>0.8)的目标数量: {len(high_conf_boxes)}")
    
    return results


def example_with_custom_config():
    """使用自定义配置"""
    print("\n" + "=" * 50)
    print("自定义配置示例")
    print("=" * 50)
    
    # 创建检测器，使用更多自定义参数
    detector = create_detector(
        model_dir='your_model_path',
        device='GPU',
        run_mode='trt_fp16',  # 使用TensorRT FP16加速
        batch_size=8,
        trt_min_shape=320,
        trt_max_shape=1280,
        trt_opt_shape=640,
        cpu_threads=4,
        enable_mkldnn=False,
        threshold=0.6,
        output_dir='custom_output',
        use_fd_format=False
    )
    
    # 进行预测
    results = predict_images(
        detector=detector,
        image_dir='test_images/',
        run_benchmark=False,
        repeats=1,
        visual=True,
        save_results=True
    )
    
    return results


if __name__ == '__main__':
    # 运行示例（根据需要取消注释）
    
    # 基础使用
    # results = example_basic_usage()
    
    # 批量预测
    # results = example_batch_prediction()
    
    # 使用图片列表
    # results = example_with_image_list()
    
    # 切片预测
    # results = example_slice_prediction()
    
    # 只获取结果
    # results = example_return_only_results()
    
    # 自定义配置
    # results = example_with_custom_config()
    
    print("\n请取消注释上面的示例来运行相应的功能")
    print("注意：需要将 'your_model_path' 替换为实际的模型路径")
    print("     需要将图片路径替换为实际的图片路径")
