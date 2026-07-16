"""
快速开始 - 最简单的使用示例
""" 

# import paddle
# import os,sys
# sys.path.insert(0,  '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/deploy/python')

# from infer import create_detector, predict_images
import os,sys
# sys.path.insert(0,  '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/deploy/python')

# 初始化Paddle
# paddle.enable_static()

# 方法1: 预测单张图片
def simple_predict_single_image():
    # 创建检测器
    detector = create_detector(
        model_dir='/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2/train2-best/infer/yolo3',  # 替换为你的模型路径
        device='GPU',  # 或 'GPU'
        threshold=0.5
    )
    
    # 预测并返回结果
    results = predict_images(
        detector=detector,
        image_file='/data/tensorflow/kath/Data/OCR/Tax/test/TaxDoc/main_48_page1.jpg',  # 替换为你的图片路径
        visual=True,  # 保存可视化结果到output目录
        save_results=True  # 保存JSON结果
    )
    
    # 打印结果
    print(f"检测到 {len(results['boxes'])} 个目标")
    
    # 遍历每个检测框
    for box in results['boxes']:
        class_id, score, x_min, y_min, x_max, y_max = box
        print(f"类别: {int(class_id)}, 置信度: {score:.3f}")
    
    return results


# 方法2: 预测一个目录中的所有图片
def simple_predict_directory():
    detector = create_detector(
        model_dir='your_model_path',
        device='CPU',
        threshold=0.5
    )
    
    results = predict_images(
        detector=detector,
        image_dir='test_images/',  # 替换为你的图片目录
        visual=True,
        save_results=True
    )
    
    print(f"处理了 {len(results['boxes_num'])} 张图片")
    print(f"总共检测到 {len(results['boxes'])} 个目标")
    
    return results


# 方法3: 预测指定的多张图片
def simple_predict_image_list():
    detector = create_detector(
        model_dir='your_model_path',
        device='CPU',
        threshold=0.5
    )
    
    # 指定要预测的图片
    my_images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    
    results = predict_images(
        detector=detector,
        image_list=my_images,
        visual=False,  # 不保存可视化
        save_results=False  # 不保存JSON，只返回结果
    )
    
    return results

def layout_simple_pic(model_dir, img_dir, th):
    import paddle
    from infer import run_detector_from_config

    paddle.enable_static()

    config = {
        'model_dir': model_dir,
        'device': 'GPU',
        'image_file': img_dir,
        'threshold': th,
        'save_images': True,
        'save_results': True,
        'output_dir': './output/',
    }
    results = run_detector_from_config(config)
    return results



def layout_file_pic(model_dir, img_dir, th):
    import paddle
    from infer import run_detector_from_config

    paddle.enable_static()

    config = {
        'model_dir': model_dir,
        'device': 'GPU',
        'image_dir': img_dir,
        'threshold': th,
        'save_images': True,
        'save_results': True,
        'output_dir': './output/',
    }
    results = run_detector_from_config(config)
    return results
    
    

if __name__ == '__main__':
    # 选择一个方法运行
    # results = simple_predict_single_image()
    # results = simple_predict_directory()
    # results = simple_predict_image_list()
    # Func2
    import paddle
    from infer import run_detector_from_config

    paddle.enable_static()

    config = {
        'model_dir': '/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2/train2-best/infer/yolo3',
        'device': 'GPU',
        'image_file': '/data/tensorflow/kath/Data/OCR/Tax/test/TaxDoc/main_48_page1.jpg',
        'threshold':0.5,
        'save_images': True,
        'save_results': True,
        'output_dir': './output/',
    }
    results = run_detector_from_config(config)
    print(results)
