"""
此文件用于使用者添加自己的分析方法
1. 在此文件中以函数的形式编写分析方法，参变量需要包含名称及类型，
如果函数中包含参数，需要提供初始值，GUI中，初始值会以默认值的形式体现，
未提供初始值将会在GUI中以被分析对象的数据属性为初始值，并提供“load”按钮
2. 在同级菜单的“__init__.py”文件中暴露该函数接口即可
"""
import numpy as np


# 以下是示例
def array_sum(
    array_1: np.ndarray,
    array_2: np.ndarray,
    weight_1: float = 0.5,
    weight_2: float = 0.5,
):
    """自定义分析示例, 计算两个矩阵加权之和, 默认权重为0.5"""

    result = array_1 * weight_1 + array_2 * weight_2
    return result
