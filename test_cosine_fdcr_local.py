"""
本地验证脚本：检查 CosineFDCR 是否设置了 training.py 需要的所有属性

直接运行验证，无需训练：
    python test_cosine_fdcr_local.py
"""

import sys
import torch
import numpy as np


def test_cosine_fdcr():
    """模拟 training.py 对 CosineFDCR 的调用，检查必需属性"""
    
    print("=" * 60)
    print("CosineFDCR 本地验证")
    print("=" * 60)
    
    # 模拟导入
    try:
        from Server.CosineFDCR import CosineFDCR_Head
        print("✅ 导入成功")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 检查类属性
    print("\n检查类结构...")
    
    required_methods = ['server_update', 'compute_score', '_build_slice_indices']
    for method in required_methods:
        if hasattr(CosineFDCR_Head, method):
            print(f"  ✅ 方法 {method} 存在")
        else:
            print(f"  ❌ 方法 {method} 缺失")
            return False
    
    # 检查 server_update 源代码中是否设置了关键属性
    print("\n检查 server_update 中的属性设置...")
    import inspect
    source = inspect.getsource(CosineFDCR_Head.server_update)
    
    required_attrs = ['self.div_score', 'self.aggregation_weight']
    for attr in required_attrs:
        if attr in source:
            print(f"  ✅ {attr} 已设置")
        else:
            print(f"  ❌ {attr} 未设置 - training.py 需要此属性")
            return False
    
    # 检查继承的属性
    print("\n检查继承链...")
    mro = CosineFDCR_Head.__mro__
    print(f"  继承链: {' -> '.join([c.__name__ for c in mro[:4]])}")
    
    # 检查 compute_score 支持的模式
    print("\n检查 compute_score 模式...")
    source_score = inspect.getsource(CosineFDCR_Head.compute_score)
    modes = ['baseline', 'cos', 'nmse', 'unit']
    for mode in modes:
        if f"mode == '{mode}'" in source_score:
            print(f"  ✅ 模式 '{mode}' 已支持")
        else:
            print(f"  ⚠️ 模式 '{mode}' 可能不支持")
    
    print("\n" + "=" * 60)
    print("✅ 所有检查通过！可以同步到服务器运行")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_cosine_fdcr()
    sys.exit(0 if success else 1)
