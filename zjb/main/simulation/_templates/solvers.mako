<%def name="_variables(variables)">
    % for name, variable in variables.items():
${name} = ${variable.expression}
    % endfor
</%def>

<%def name="_diff_states(state_variables)">
    % for name, state in state_variables.items():
_d${name} = ${state.expression}
    % endfor
</%def>

<%def name="euler(solver, model, env)">
${_variables(model.coupling_variables)}
${_variables(model.transient_variables)}
${_diff_states(model.state_variables)}
% if solver.noises:
    <%
        for name, value in solver.noises.items():
            env[f'__solver_noise_{name}'] = value
    %>
    % for name in model.state_variables:
        %if name in solver.noises:
${name} += _d${name} * __dt + __solver_noise_${name} * np.sqrt(__dt) * np.random.randn(__nr)
        % else:
${name} += _d${name} * __dt
        % endif
    % endfor
% else:
    % for name in model.state_variables:
${name} += _d${name} * __dt
    % endfor
% endif
</%def>