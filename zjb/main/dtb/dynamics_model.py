import json
from pathlib import Path

import brainpy as bp
import brainpy.math as bm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from traits.api import (
    Any,
    Dict,
    Expression,
    Float,
    HasPrivateTraits,
    HasRequiredTraits,
    List,
    Str,
    Subclass,
)
from zjb._traits.types import Instance
from zjb.dos.data import Data


class HasExpression(HasPrivateTraits, HasRequiredTraits):
    """表达式的基类"""
    expression = Expression(required=True)


class StateVariable(HasExpression):
    """状态变量类，继承自HasExpression。"""
    pass


class CouplingVariable(HasExpression):
    """耦合变量类，继承自HasExpression。"""
    pass


class TransientVariable(HasExpression):
    """瞬态变量类，继承自HasExpression。"""
    pass


class DynamicsModel(Data):
    """
    动力学模型类，用于存储和处理动力学相关数据。

    Attributes
    ---------
    name : Str
        动力学模型的名称。
    state_variables : Dict(Str, Instance(StateVariable))
        状态变量字典，包括每个状态变量的实例。
    coupling_variables : Dict(Str, Instance(CouplingVariable))
        耦合变量字典，包括每个耦合变量的实例。
    transient_variables : Dict(Str, Instance(TransientVariable))
        瞬态变量字典，包括每个瞬态变量的实例。
    parameters : Dict(Str, Float)
        参数字典，包括每个参数的数值。
    docs : Dict(Str, Str)
        文档字典，包含模型相关的文档说明。
    references : List(Str)
        参考文献列表，包含与模型相关的参考文献。
    """
    name = Str()

    state_variables = Dict(Str, Instance(StateVariable))

    coupling_variables = Dict(Str, Instance(CouplingVariable))

    transient_variables = Dict(Str, Instance(TransientVariable))

    parameters = Dict(Str, Float)

    docs = Dict(Str, Str)

    references = List(Str)

    @classmethod
    def from_file(cls, filename: str):
        """
        从JSON文件中加载动力学模型数据，创建并返回一个模型实例。

        Parameters
        ----------
        filename : str
            要读取的文件名。

        Returns
        -------
        DynamicsModel
            根据JSON文件中的数据创建的动力学模型实例。
        """
        with open(filename) as f:
            obj = json.load(f)
        obj["state_variables"] = {
            name: StateVariable(**state)
            for name, state in obj["state_variables"].items()
        }
        obj["coupling_variables"] = {
            name: CouplingVariable(**coupling)
            for name, coupling in obj["coupling_variables"].items()
        }
        obj["transient_variables"] = {
            name: TransientVariable(**transient)
            for name, transient in obj["transient_variables"].items()
        }
        return cls(**obj)

    @classmethod
    def from_name(cls, name: str):
        """
        根据动力学模型的名称加载相应的JSON文件，并创建模型实例。

        Parameters
        ----------
        name : str
            动力学模型的名称。

        Returns
        -------
        DynamicsModel
            根据指定名称创建的动力学模型实例。
        """
        file = Path(__file__).parent / "_dynamics_models" / f"{name}.json"
        if not file.exists():
            raise ValueError(f"{name} not found.")
        return cls.from_file(str(file))

    @classmethod
    def list_names(cls):
        """
        列出可用的动力学模型名称。

        Returns
        -------
        list
            包含所有可用动力学模型名称的列表。
        """
        return [
            file.stem
            for file in (Path(__file__).parent / "_dynamics_models").iterdir()
            if file.is_file() and file.suffix == ".json"
        ]

    def phase_plane_analyse(
        self,
        target_vars: dict,
        fixed_vars: dict,
        resolutions: dict,
        trajectory: dict,
        trajectory_duration: float,
        show=False,
    ):
        """
        对动力学模型进行相平面分析。

        Parameters
        ----------
        target_vars : dict
            目标的状态变量及其范围。
        fixed_vars : dict
            固定的状态变量及值。
        resolutions : dict
            状态变量相平面分析的步长。
        trajectory : dict
            初始轨迹的设定。
        trajectory_duration : float
            轨迹持续的时间。
        show : bool, optional
            是否显示分析结果，默认为False。

        Returns
        -------
        相平面分析的结果的展示图像。
        """
        pp_analyse = PhasePlaneFunc()
        pp_analyse.model = self
        pp_analyse.target_vars = target_vars
        pp_analyse.fixed_vars = fixed_vars
        pp_analyse.resolutions = resolutions
        pp_analyse.trajectory = trajectory
        pp_analyse.trajectory_duration = trajectory_duration

        return pp_analyse(show)

    def bifurcation_analyse(
        self,
        target_vars: dict,
        fixed_vars: dict,
        target_pars: dict,
        resolutions: dict,
        show=False,
    ):
        """
        对动力学模型进行分岔分析。

        Parameters
        ----------
        target_vars : dict
            目标状态变量的范围。
        fixed_vars : dict
            固定的状态变量的值。
        target_pars : dict
            目标参数的范围。
        resolutions : dict
            分析目标参数时的步长。
        show : bool, optional
            是否显示分析结果，默认为False。

        Returns
        -------
        分岔分析结果的展示图s
        """
        bifurcation_analyse = BifurcationFunc()
        bifurcation_analyse.model = self
        bifurcation_analyse.target_vars = target_vars
        bifurcation_analyse.fixed_vars = fixed_vars
        bifurcation_analyse.target_pars = target_pars
        bifurcation_analyse.resolutions = resolutions

        return bifurcation_analyse(show)


class DefineExpression:
    """Model表达式"""

    @classmethod
    def var_expression(cls, model: DynamicsModel, var_name: str):
        """
        根据指定的动力学模型和状态变量名，生成该状态变量的表达式。

        Parameters
        ----------
        model : DynamicsModel
            动力学模型实例。
        var_name : str
            要生成表达式的变量名。

        Returns
        -------
        str:
           动态成的状态变量表达式的字符串。
        """
        # state_variables
        states = ",".join(model.state_variables.keys())

        variables = states
        variables = variables.replace(f"{var_name},", "").replace(f",{var_name}", "")
        variables = var_name + ",t," + variables

        # transient_variables
        # 瞬态变量通常是状态变量表达式的一部分，因此需要先计算瞬态态变量，再计算状态变量
        # TODO: 特定的状态变量往往依赖部分瞬态变量，分析状态变量的表达式可以确定哪些状态变量是不必要的
        transient_expressions = "\n    ".join(
            f"{name}={var.expression}"
            for name, var in model.transient_variables.items()
        )

        # parameters
        parameters = ""
        for para in list(model.parameters.keys()):
            value = model.parameters[para]
            parameters += para + "=" + f"{value}, "
        # 'd' + list(model.state_variables.keys())[0] + f'({states}, {str_para})'

        coupling = ""
        for coup in list(model.coupling_variables.keys()):
            value = model.coupling_variables[coup]
            coupling += coup + "=" + "0, "

        expression = model.state_variables[var_name].expression

        return f"""
def d{var_name}({variables}, {parameters} {coupling}):
    {transient_expressions}
    return {expression}
"""


class ModelTypesetter(bp.DynamicalSystem):
    """Model的公式排版器，提取解析式"""

    def __init__(self, model: DynamicsModel, method="exp_auto"):
        super(ModelTypesetter, self).__init__()

        # parameter
        for para in list(model.parameters.keys()):
            value = model.parameters[para]
            setattr(self, para, value)

        # variables
        for var in list(model.state_variables.keys()):
            setattr(
                self,
                var,
                bm.Variable(bm.zeros(1)),
            )

        # functions
        define_expression = DefineExpression()

        for var in list(model.state_variables.keys()):
            env = {"np": bm}
            exec(
                compile(
                    define_expression.var_expression(model, var), "<string>", "exec"
                ),
                env,
            )
            var_fun = env["d" + var]
            setattr(self, "int_" + var, bp.odeint(var_fun, method=method))


class PhasePlaneAnalyzer(ModelTypesetter):
    """相平面分析器"""
    pass


class BifurcationAnalyzer(ModelTypesetter):
    """分岔分析器"""
    pass


class Plots(Data):
    """专为用matplotlib展示的分析结果设计的数据类型"""

    figure = Instance(Figure)

    name = Str()


class PhasePlane_2D(bp.analysis.PhasePlane2D):
    """相平面2D分析类，继承自brainpy.analysis.PhasePlane2D。"""
    def show_figure_data(self):
        """
        获取当前matplotlib图形的Figure对象。

        Returns
        -------
        Figure
            当前的matplotlib图形对象。
        """
        global plt
        if plt is None:
            from matplotlib import pyplot as plt
        fig = plt.gcf()
        return fig


class Bifurcation_2D(bp.analysis.Bifurcation2D):
    """分岔2D分析类，继承自brainpy.analysis.Bifurcation2D。"""
    def show_figure_data(self, var):
        """
        获取相应matplotlib图形的Figure对象。

        Parameters
        ----------
        var : str
            变量名，用于标识图形。

        Returns
        -------
        Figure
            对应变量的matplotlib图形对象。
        """
        global plt
        if plt is None:
            from matplotlib import pyplot as plt
        fig = plt.figure(var)
        return fig


class PhasePlanePlots(Plots):
    """
    相平面绘图类，用于存储和展示相平面分析结果。

    Attributes
    ----------
    dynamicsModel : Instance(DynamicsModel)
        关联的动力学模型实例。
    target_vars : Dict
        目标状态变量的范围。
    fixed_vars : Dict
        固定状态变量的值。
    resolutions : Dict
        状态变量分析的分辨率（步长）。
    trajectory : Dict
        初始轨迹的设定。
    trajectory_duration : Float
        轨迹持续的时间。
    fixed_points : Any
        定点坐标，或无定点。
    """
    dynamicsModel = Instance(DynamicsModel)

    target_vars = Dict()

    fixed_vars = Dict()

    resolutions = Dict()

    trajectory = Dict()

    trajectory_duration = Float()

    fixed_points = Any()  # 定点坐标, 或无定点


class BifurcationPlots(Plots):
    """
    分岔绘图类，用于存储和展示分岔分析结果。

    Attributes
    ----------
    figure2 : Instance(Figure)
        2D分岔分析产生的第二副分岔图。
    dynamicsModel : Instance(DynamicsModel)
        关联的动力学模型实例。
    target_vars : Dict
        目标状态变量的范围。
    fixed_vars : Dict
        固定状态变量的值。
    target_pars : Dict
        目标参数的范围。
    resolutions : Dict
        分析目标参数的分辨率（步长）。
    """
    figure2 = Instance(Figure)  # 2D分岔分析会产生2副分岔图

    dynamicsModel = Instance(DynamicsModel)

    target_vars = Dict()

    fixed_vars = Dict()

    target_pars = Dict()

    resolutions = Dict()


class PhasePlaneFunc(HasPrivateTraits):
    """
    对动力学模型进行相平面分析的类。

    Attributes
    ----------
    model : DynamicsModel
        要分析的动力学模型实例。
    ppanalyzer : PhasePlaneAnalyzer
        相平面分析器的实例。
    target_vars : Dict
        目标状态变量的范围。
    fixed_vars : Dict
        固定状态变量的值。
    resolutions : Dict
        状态变量分析的分辨率。
    trajectory : Dict
        初始轨迹的设定。
    trajectory_duration : Float
        轨迹持续的时间。
    """

    model: DynamicsModel = Instance(DynamicsModel, input=True)

    ppanalyzer: PhasePlaneAnalyzer = Subclass(PhasePlaneAnalyzer, ModelTypesetter)

    target_vars = Dict()

    fixed_vars = Dict()

    resolutions = Dict()

    trajectory = Dict()

    trajectory_duration = Float()

    def __call__(self, show=False):
        """
        执行相平面分析并返回结果。

        Parameters
        ----------
        show : bool, optional
            是否显示分析结果，默认为False。

        Returns
        -------
        PhasePlanePlots
            相平面分析的结果，包括图形和相关数据。
        """
        plt.cla()
        bp.math.enable_x64()
        transmodel = self.model
        model = self.ppanalyzer(transmodel)
        analyzer = PhasePlane_2D(
            model,
            target_vars=self.target_vars,
            fixed_vars=self.fixed_vars,
            resolutions=self.resolutions,
        )
        analyzer.plot_nullcline(x_style={"fmt": "-"}, y_style={"fmt": "-"})

        if show:
            analyzer.plot_vector_field(show=True)
        else:
            analyzer.plot_vector_field(show=False)

        fixed_points = analyzer.plot_fixed_point(with_return=True)

        if fixed_points is not None:
            fixed_points = np.around(fixed_points, 3)
        else:
            fixed_points = "None"

        analyzer.plot_trajectory(self.trajectory, duration=self.trajectory_duration)

        # show the phase plane figure
        # analyzer.show_figure()

        return PhasePlanePlots(
            name="phase plane results",
            figure=analyzer.show_figure_data(),
            dynamicsModel=self.model,
            target_vars=self.target_vars,
            fixed_vars=self.fixed_vars,
            resolutions=self.resolutions,
            trajectory=self.trajectory,
            trajectory_duration=self.trajectory_duration,
            fixed_points=fixed_points,
        )


class BifurcationFunc(HasPrivateTraits):
    """
    对动力学模型进行分岔分析的类。

    Attributes
    ----------
    model : DynamicsModel
        要分析的动力学模型实例。
    bifurcation_analyzer : BifurcationAnalyzer
        分岔分析器的实例。
    target_vars : Dict
        目标状态变量的范围。
    fixed_vars : Dict
        固定状态变量的值。
    target_pars : Dict
        目标参数的范围。
    resolutions : Dict
        分析目标参数的分辨率（步长）。
    """

    model: DynamicsModel = Instance(DynamicsModel, input=True)

    bifurcation_analyzer: BifurcationAnalyzer = Subclass(
        BifurcationAnalyzer, ModelTypesetter
    )

    target_vars = Dict()

    fixed_vars = Dict()

    target_pars = Dict()

    resolutions = Dict()

    def __call__(self, show=False):
        """
        执行分岔分析并返回结果。

        Parameters
        ----------
        show : bool, optional
            是否显示分析结果，默认为False。

        Returns
        -------
        BifurcationPlots
            分岔分析的结果，包括图形和相关数据。
        """
        plt.cla()
        bp.math.enable_x64()
        model = self.bifurcation_analyzer(self.model)
        analyzer = Bifurcation_2D(
            model,
            target_vars=self.target_vars,
            fixed_vars=self.fixed_vars,
            target_pars=self.target_pars,
            resolutions=self.resolutions,
        )
        if show:
            final_fps, final_pars, jacobians = analyzer.plot_bifurcation(
                with_return=True, num_rank=10, show=True
            )
        else:
            final_fps, final_pars, jacobians = analyzer.plot_bifurcation(
                with_return=True, num_rank=10, show=False
            )

        fig1 = analyzer.show_figure_data(list(self.target_vars.keys())[0])
        fig2 = analyzer.show_figure_data(list(self.target_vars.keys())[1])

        return BifurcationPlots(
            name="Bifurcation analysis results",
            figure=fig1,
            figure2=fig2,
            dynamicsModel=self.model,
            target_vars=self.target_vars,
            fixed_vars=self.fixed_vars,
            target_pars=self.target_pars,
            resolutions=self.resolutions,
        )


if __name__ == "__main__":
    # 测试动力学分析

    # 创建测试模型
    dynamic = DynamicsModel()

    dynamic.state_variables = {
        "x": StateVariable(expression="ax2y2 * x - omega * y + Gx"),
        "y": StateVariable(expression="ax2y2 * y + omega * x + Gy"),
    }
    dynamic.coupling_variables = {
        "Gx": CouplingVariable(expression="__C @ x - __C_1 * x"),
        "Gy": CouplingVariable(expression="__C @ y - __C_1 * y"),
    }
    dynamic.transient_variables = {
        "ax2y2": TransientVariable(expression="a - x * x - y * y")
    }
    dynamic.parameters = {
        "a": -0.1,
        "omega": 1.0,
    }

    # 相平面分析功能测试
    target_vars = {"x": [-2, 2], "y": [-2, 2]}
    resolutions = {"x": 0.1, "y": 0.1}
    trajectory = {"x": [0], "y": [0]}
    trajectory_duration = 5

    fixed_vars = {}

    dynamic.phase_plane_analyse(
        target_vars, fixed_vars, resolutions, trajectory, trajectory_duration, show=True
    )

    target_vars = {"x": [-2, 2], "y": [-2, 2]}
    fixed_vars = {}
    target_pars = {"a": [-5, 5], "omega": [-5, 5]}
    resolutions = {"a": 0.1, "omega": 0.1}

    # 分岔分析功能测试
    dynamic.bifurcation_analyse(
        target_vars, fixed_vars, target_pars, resolutions, show=True
    )
