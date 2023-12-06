<%!
    def indent(text):
        return text.replace("\n", "\n    ")
    def indent2(text):
        return text.replace("\n", "\n        ")
%>
<%
    from zjb.main.simulation.simulator import ExprParameter
%>
import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def simulator(
    __t, __dt, __C,
    ${','.join(model.state_variables)}
):
    __nt = int(__t / __dt)
    __nr =  __C.shape[0]
    __C_1 =  np.sum(__C, 1)

    % for name, state in model.state_variables.items():
    ${name} = np.ones(__nr) * ${name}
    % endfor

    % for i, monitor in enumerate(monitors):
    ${monitor.render_init(f'_m{i}', env) | indent}
    % endfor

    for __it in range(__nt):
        __ct = __it * __dt
        % for name, para in simulator.parameters.items():
            % if isinstance(para, ExprParameter):
        ${name} = ${para.expression}
            % endif
        % endfor

        ${solver.render(model, env) | indent2}

        % for i, monitor in enumerate(monitors):
        ${monitor.render_sample(f'_m{i}', env) | indent2}
        % endfor

    return (
        ${','.join(model.state_variables)},
    ),(
        ${','.join(f"_m{i}" for i in range(len(monitors)))},
    )