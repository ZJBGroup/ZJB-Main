from traits.api import (
    Dict,
    Expression,
    Float,
    HasPrivateTraits,
    HasRequiredTraits,
    List,
    Str,
    Union,
)

from zjb._traits.types import Instance
from zjb.dos.data import Data
from zjb.main.trait_types import FloatVector


class HasExpression(HasPrivateTraits, HasRequiredTraits):
    expression = Expression(required=True)


class StateVariable(HasExpression):
    pass


class CouplingVariable(HasExpression):
    pass


class TransientVariable(HasExpression):
    pass


class DynamicModel(Data):
    name = Str()

    state_variables = Dict(Str, Instance(StateVariable))

    coupling_variables = Dict(Str, Instance(CouplingVariable))

    transient_variables = Dict(Str, Instance(TransientVariable))

    parameters = Dict(Str, Union(Float, FloatVector))

    references = List(Str)
