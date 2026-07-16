"""
简单验证 - 检查API函数是否已正确添加到 infer.py
不需要导入paddle
"""

import ast
import os

def check_infer_file():
    """检查 infer.py 文件中的函数定义"""
    
    infer_file = 'infer.py'
    
    if not os.path.exists(infer_file):
        print(f"✗ 找不到文件: {infer_file}")
        return False
    
    with open(infer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查新增的函数是否存在
    functions_to_check = [
        'create_detector',
        'predict_images', 
        'predict_images_slice'
    ]
    
    print("=" * 60)
    print("检查 infer.py 中的API函数")
    print("=" * 60)
    
    all_found = True
    
    for func_name in functions_to_check:
        # 检查函数定义
        if f'def {func_name}(' in content:
            print(f"✓ 找到函数: {func_name}")
            
            # 查找函数的文档字符串
            func_pos = content.find(f'def {func_name}(')
            if func_pos != -1:
                # 查找下一个 """
                docstring_start = content.find('"""', func_pos)
                if docstring_start != -1 and docstring_start - func_pos < 200:
                    docstring_end = content.find('"""', docstring_start + 3)
                    if docstring_end != -1:
                        docstring = content[docstring_start+3:docstring_end].strip()
                        first_line = docstring.split('\n')[0]
                        print(f"  文档: {first_line[:50]}...")
        else:
            print(f"✗ 未找到函数: {func_name}")
            all_found = False
    
    # 检查原有类是否存在
    print("\n检查向后兼容性:")
    original_items = ['class Detector', 'def get_test_images', 'def visualize', 'def main']
    
    for item in original_items:
        if item in content:
            print(f"✓ 保留原有: {item}")
        else:
            print(f"✗ 缺失: {item}")
            all_found = False
    
    # 统计代码行数变化
    lines = content.split('\n')
    print(f"\n文件总行数: {len(lines)}")
    
    return all_found


def check_example_files():
    """检查示例文件是否已创建"""
    
    print("\n" + "=" * 60)
    print("检查示例文件")
    print("=" * 60)
    
    example_files = [
        'API_USAGE.md',
        'quick_start.py',
        'infer_api_example.py',
        'practical_example.py',
        'CHANGES_SUMMARY.md'
    ]
    
    all_exist = True
    
    for filename in example_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename} ({size:,} 字节)")
        else:
            print(f"✗ {filename} (不存在)")
            all_exist = False
    
    return all_exist


def show_usage_example():
    """显示使用示例"""
    
    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    
    example = """
import paddle
from infer import create_detector, predict_images

# 初始化
paddle.enable_static()

# 创建检测器
detector = create_detector(
    model_dir='your_model_path',
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
"""
    
    print(example)


def main():
    """主函数"""
    
    print("🔍 验证 infer.py API 改造\n")
    
    # 检查文件
    code_ok = check_infer_file()
    files_ok = check_example_files()
    
    # 显示使用示例
    show_usage_example()
    
    # 总结
    print("=" * 60)
    print("验证总结")
    print("=" * 60)
    
    if code_ok and files_ok:
        print("✓ 所有检查通过！")
        print("\n✅ API改造成功！")
        print("\n📚 下一步:")
        print("1. 查看 CHANGES_SUMMARY.md 了解修改内容")
        print("2. 查看 API_USAGE.md 学习使用方法")
        print("3. 运行 quick_start.py 进行实际测试（需要修改模型路径）")
        print("4. 参考 infer_api_example.py 查看更多示例")
        print("5. 参考 practical_example.py 了解如何集成到工作流")
    else:
        print("⚠ 部分检查未通过")
        if not code_ok:
            print("  - 代码检查失败")
        if not files_ok:
            print("  - 示例文件缺失")
    
    return code_ok and files_ok


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
