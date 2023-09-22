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

<%def name="euler(model)">
${_variables(model.coupling_variables)}
${_variables(model.transient_variables)}
${_diff_states(model.state_variables)}
% for name in model.state_variables:
${name} += _d${name} * __dt
% endfor
</%def>