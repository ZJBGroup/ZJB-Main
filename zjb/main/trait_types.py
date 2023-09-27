from traits.api import Any, Array

IntVector = Array(dtype=int, shape=(None,))
FloatVector = Array(dtype=float, shape=(None,))
StrVector = Array(dtype=str, shape=(None,))
BoolVector = Array(dtype=bool, shape=(None,))

TraitAny = Any
