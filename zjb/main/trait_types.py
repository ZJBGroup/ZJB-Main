from traits.api import Array

IntVector = Array(dtype=int, shape=(None,))
FloatVector = Array(dtype=float, shape=(None,))
StrVector = Array(dtype=str, shape=(None,))
BoolVector = Array(dtype=bool, shape=(None,))

RequiredIntVector = Array(dtype=int, shape=(None,), required=True)
RequiredFloatVector = Array(dtype=float, shape=(None,), required=True)
RequiredStrVector = Array(dtype=str, shape=(None,), required=True)
RequiredBoolVector = Array(dtype=bool, shape=(None,), required=True)
