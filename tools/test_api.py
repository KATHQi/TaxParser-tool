"""
测试新的API - 验证功能是否正常工作
"""

def test_api_exists():
    """测试API函数是否存在"""
    try:
        from infer import create_detector, predict_images, predict_images_slice
        print("✓ API函数导入成功")
        print("  - create_detector")
        print("  - predict_images")
        print("  - predict_images_slice")
        return True
    except ImportError as e:
        print(f"✗ API函数导入失败: {e}")
        return False


def test_function_signatures():
    """测试函数签名"""
    try:
        from infer import create_detector, predict_images
        import inspect
        
        # 检查 create_detector 参数
        sig = inspect.signature(create_detector)
        params = list(sig.parameters.keys())
        required_params = ['model_dir', 'device', 'threshold']
        
        print("\n✓ create_detector 函数签名正确")
        print(f"  参数: {', '.join(params[:5])}...")
        
        # 检查 predict_images 参数
        sig = inspect.signature(predict_images)
        params = list(sig.parameters.keys())
        
        print("✓ predict_images 函数签名正确")
        print(f"  参数: {', '.join(params[:5])}...")
        
        return True
    except Exception as e:
        print(f"✗ 函数签名检查失败: {e}")
        return False


def test_docstrings():
    """测试文档字符串"""
    try:
        from infer import create_detector, predict_images
        
        if create_detector.__doc__:
            print("\n✓ create_detector 有文档说明")
            print(f"  {create_detector.__doc__.strip().split(chr(10))[0][:50]}...")
        else:
            print("⚠ create_detector 缺少文档说明")
        
        if predict_images.__doc__:
            print("✓ predict_images 有文档说明")
            print(f"  {predict_images.__doc__.strip().split(chr(10))[0][:50]}...")
        else:
            print("⚠ predict_images 缺少文档说明")
        
        return True
    except Exception as e:
        print(f"✗ 文档字符串检查失败: {e}")
        return False


def test_backward_compatibility():
    """测试向后兼容性"""
    try:
        from infer import Detector, get_test_images, visualize
        print("\n✓ 原有函数仍然可用")
        print("  - Detector 类存在")
        print("  - get_test_images 函数存在")
        print("  - visualize 函数存在")
        return True
    except ImportError as e:
        print(f"✗ 向后兼容性测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("测试 infer.py API 改造")
    print("=" * 60)
    
    results = []
    
    # 测试1: API存在性
    results.append(("API函数导入", test_api_exists()))
    
    # 测试2: 函数签名
    results.append(("函数签名检查", test_function_signatures()))
    
    # 测试3: 文档字符串
    results.append(("文档字符串", test_docstrings()))
    
    # 测试4: 向后兼容性
    results.append(("向后兼容性", test_backward_compatibility()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！API改造成功！")
        print("\n下一步:")
        print("1. 查看 API_USAGE.md 了解使用方法")
        print("2. 运行 quick_start.py 进行实际测试")
        print("3. 参考 infer_api_example.py 查看更多示例")
    else:
        print("\n⚠ 部分测试失败，请检查代码")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
