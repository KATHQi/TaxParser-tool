"""
实际应用示例 - 将检测结果集成到你的工作流中
"""

import paddle
import json
import os
from infer import create_detector, predict_images

# 初始化
paddle.enable_static()


def detect_and_process(model_dir, image_path, output_dir='output'):
    """
    检测图像并处理结果的完整流程
    
    Args:
        model_dir: 模型目录
        image_path: 图像路径或目录
        output_dir: 输出目录
        
    Returns:
        dict: 处理后的结果
    """
    # 1. 创建检测器
    print(f"正在加载模型: {model_dir}")
    detector = create_detector(
        model_dir=model_dir,
        device='CPU',  # 根据实际情况改为 'GPU'
        threshold=0.5,
        output_dir=output_dir
    )
    
    # 2. 执行检测
    print(f"正在检测: {image_path}")
    if os.path.isfile(image_path):
        results = predict_images(
            detector=detector,
            image_file=image_path,
            visual=True,
            save_results=False  # 我们自己处理结果
        )
    else:
        results = predict_images(
            detector=detector,
            image_dir=image_path,
            visual=True,
            save_results=False
        )
    
    # 3. 处理结果
    processed_results = process_detection_results(results)
    
    # 4. 保存自定义格式的结果
    save_custom_results(processed_results, output_dir)
    
    return processed_results


def process_detection_results(results):
    """
    将原始检测结果处理成更易用的格式
    
    Args:
        results: predict_images返回的结果
        
    Returns:
        list: 处理后的结果列表
    """
    processed = []
    boxes = results['boxes']
    boxes_num = results['boxes_num']
    
    start_idx = 0
    for img_idx, num_boxes in enumerate(boxes_num):
        img_result = {
            'image_index': img_idx,
            'num_detections': int(num_boxes),
            'detections': []
        }
        
        # 获取该图片的所有检测框
        for i in range(num_boxes):
            box = boxes[start_idx + i]
            
            detection = {
                'class_id': int(box[0]),
                'confidence': float(box[1]),
                'bbox': {
                    'x_min': float(box[2]),
                    'y_min': float(box[3]),
                    'x_max': float(box[4]),
                    'y_max': float(box[5]),
                    'width': float(box[4] - box[2]),
                    'height': float(box[5] - box[3])
                }
            }
            img_result['detections'].append(detection)
        
        start_idx += num_boxes
        processed.append(img_result)
    
    return processed


def save_custom_results(processed_results, output_dir):
    """
    保存自定义格式的结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON
    output_file = os.path.join(output_dir, 'detection_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_file}")
    
    # 生成统计报告
    report_file = os.path.join(output_dir, 'detection_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("检测结果统计报告\n")
        f.write("=" * 50 + "\n\n")
        
        total_detections = sum(r['num_detections'] for r in processed_results)
        f.write(f"总图片数: {len(processed_results)}\n")
        f.write(f"总检测数: {total_detections}\n")
        f.write(f"平均每张图片检测数: {total_detections / len(processed_results):.2f}\n\n")
        
        # 按类别统计
        class_counts = {}
        for result in processed_results:
            for det in result['detections']:
                class_id = det['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        f.write("各类别检测数量:\n")
        for class_id, count in sorted(class_counts.items()):
            f.write(f"  类别 {class_id}: {count}\n")
    
    print(f"统计报告已保存到: {report_file}")


def filter_detections_by_confidence(results, min_confidence=0.7):
    """
    筛选高置信度的检测结果
    """
    filtered = []
    for img_result in results:
        filtered_img = {
            'image_index': img_result['image_index'],
            'detections': [
                det for det in img_result['detections'] 
                if det['confidence'] >= min_confidence
            ]
        }
        filtered_img['num_detections'] = len(filtered_img['detections'])
        filtered.append(filtered_img)
    
    return filtered


def filter_detections_by_class(results, target_classes):
    """
    筛选特定类别的检测结果
    
    Args:
        results: 处理后的结果
        target_classes: 目标类别ID列表
    """
    filtered = []
    for img_result in results:
        filtered_img = {
            'image_index': img_result['image_index'],
            'detections': [
                det for det in img_result['detections'] 
                if det['class_id'] in target_classes
            ]
        }
        filtered_img['num_detections'] = len(filtered_img['detections'])
        filtered.append(filtered_img)
    
    return filtered


def get_detection_statistics(results):
    """
    获取检测结果的统计信息
    """
    stats = {
        'total_images': len(results),
        'total_detections': 0,
        'class_distribution': {},
        'confidence_stats': {
            'min': 1.0,
            'max': 0.0,
            'mean': 0.0
        }
    }
    
    all_confidences = []
    
    for img_result in results:
        stats['total_detections'] += img_result['num_detections']
        
        for det in img_result['detections']:
            # 统计类别分布
            class_id = det['class_id']
            stats['class_distribution'][class_id] = \
                stats['class_distribution'].get(class_id, 0) + 1
            
            # 收集置信度
            all_confidences.append(det['confidence'])
    
    # 计算置信度统计
    if all_confidences:
        stats['confidence_stats']['min'] = min(all_confidences)
        stats['confidence_stats']['max'] = max(all_confidences)
        stats['confidence_stats']['mean'] = sum(all_confidences) / len(all_confidences)
    
    return stats


# ============= 使用示例 =============

def example_complete_workflow():
    """完整的检测工作流示例"""
    
    # 1. 执行检测
    results = detect_and_process(
        model_dir='your_model_path',
        image_path='test_images/',
        output_dir='detection_output'
    )
    
    # 2. 筛选高置信度结果
    high_conf_results = filter_detections_by_confidence(results, min_confidence=0.8)
    print(f"\n高置信度(>0.8)检测数: {sum(r['num_detections'] for r in high_conf_results)}")
    
    # 3. 筛选特定类别
    target_results = filter_detections_by_class(results, target_classes=[0, 1, 2])
    print(f"目标类别检测数: {sum(r['num_detections'] for r in target_results)}")
    
    # 4. 获取统计信息
    stats = get_detection_statistics(results)
    print("\n检测统计:")
    print(f"  总图片数: {stats['total_images']}")
    print(f"  总检测数: {stats['total_detections']}")
    print(f"  置信度范围: {stats['confidence_stats']['min']:.3f} - {stats['confidence_stats']['max']:.3f}")
    print(f"  平均置信度: {stats['confidence_stats']['mean']:.3f}")
    print(f"  类别分布: {stats['class_distribution']}")
    
    return results


def example_single_image_quick():
    """快速检测单张图片"""
    
    detector = create_detector(
        model_dir='your_model_path',
        device='CPU',
        threshold=0.5
    )
    
    results = predict_images(
        detector=detector,
        image_file='test.jpg',
        visual=False,
        save_results=False
    )
    
    # 直接处理结果
    for box in results['boxes']:
        class_id, conf, x1, y1, x2, y2 = box
        print(f"检测到类别 {int(class_id)}, 置信度 {conf:.3f}, "
              f"位置 ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
    
    return results


def example_batch_with_different_params():
    """使用不同参数批量检测"""
    
    # 创建一个检测器实例
    detector = create_detector(
        model_dir='your_model_path',
        device='GPU',
        batch_size=4,
        threshold=0.5
    )
    
    # 批量检测多个目录
    directories = ['dir1/', 'dir2/', 'dir3/']
    all_results = []
    
    for idx, dir_path in enumerate(directories):
        print(f"\n处理目录 {idx+1}/{len(directories)}: {dir_path}")
        
        results = predict_images(
            detector=detector,
            image_dir=dir_path,
            visual=True,
            save_results=False
        )
        
        processed = process_detection_results(results)
        all_results.extend(processed)
        
        print(f"  检测到 {sum(r['num_detections'] for r in processed)} 个目标")
    
    # 保存合并后的结果
    with open('all_detection_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return all_results


if __name__ == '__main__':
    # 运行示例
    
    # 完整工作流
    # results = example_complete_workflow()
    
    # 快速检测单张图片
    # results = example_single_image_quick()
    
    # 批量检测多个目录
    # results = example_batch_with_different_params()
    
    print("\n请取消注释上面的示例来运行")
    print("记得替换 'your_model_path' 为实际的模型路径")
