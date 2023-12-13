import numpy as np
from traits.api import Array

IntVector = Array(dtype=int, shape=(None,))
FloatVector = Array(dtype=float, shape=(None,))
StrVector = Array(dtype=str, shape=(None,))
BoolVector = Array(dtype=bool, shape=(None,))

RequiredIntVector = Array(dtype=int, shape=(None,), required=True)
RequiredFloatVector = Array(dtype=float, shape=(None,), required=True)
RequiredStrVector = Array(dtype=str, shape=(None,), required=True)
RequiredBoolVector = Array(dtype=bool, shape=(None,), required=True)


class ArrayLike(Array):
    """ArrayLike是一个numpy数组, 或支持__array__协议的类型, 或列表和元组.

    相较于traits库提供的Array扩展了对支持`__array__`协议的类型的支持

    对于支持`__array__`协议的类型(不包括ndarray,list和tuple), ArrayLike会
    保存其原始对象(这是为了能利用Data的引用保存), 因此需要注意在使用ArrayLike
    的值之前可能要调用`np.asarray`将数据转换为合适的ndarray类型。
    """

    def validate(self, object, name, value):
        # 使用父类验证值
        if hasattr(value, "__array__"):
            # 提前调用asarray以绕过父类的限制
            super().validate(object, name, np.asarray(value))
            # 返回转换前的值
            return value
        else:
            return super().validate(object, name, value)


class CArrayLike(Array):
    """CArrayLike是一个numpy数组, 或支持__array__协议的类型, 或列表和元组

    相比与ArrayLike, CArrayLike的值会被强制转换为ndarray类型, 而不会保留原始值
    """

    def validate(self, object, name, value):
        if hasattr(value, "__array__"):
            # 提前调用asarray已绕过父类的限制
            value = np.asarray(value)
        # 使用父类验证值并返回
        return super().validate(object, name, value)
