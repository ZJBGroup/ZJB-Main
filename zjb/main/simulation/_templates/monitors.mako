<%def name="raw_init(name, monitor, env)">
${name} = np.zeros((__nt, __nr))
</%def>

<%def name="raw_sample(name, monitor, env)">
${name}[__it] = ${monitor.expression}
</%def>

<%def name="sub_sample_init(name, monitor, env)">
${name} = np.zeros((int(__nt / ${monitor.sample_interval}), __nr))
${name}_i = 0
</%def>

<%def name="sub_sample_sample(name, monitor, env)">
if __it % ${monitor.sample_interval} == ${monitor.sample_interval} - 1:
    ${name}[${name}_i] = ${monitor.expression}
    ${name}_i += 1
</%def>

<%def name="temporal_average_init(name, monitor, env)">
${name} = np.zeros((int(__nt / ${monitor.sample_interval}), __nr))
${name}_temp = 0
${name}_i = 0
</%def>

<%def name="temporal_average_sample(name, monitor, env)">
${name}_temp = ${monitor.expression}
if __it % ${monitor.sample_interval} == ${monitor.sample_interval} - 1:
    ${name}[${name}_i] = ${name}_temp / ${monitor.sample_interval}
    ${name}_temp = 0
    ${name}_i += 1
</%def>


<%def name="bold_init(name, monitor, env)">
<%
    env[f'{name}_itaus'] = 1 / monitor.taus
    env[f'{name}_itauf'] = 1 / monitor.tauf
    env[f'{name}_itauo'] = 1 / monitor.tauo
    env[f'{name}_ialpha'] = 1 / monitor.alpha
    env[f'{name}_Eo'] = monitor.Eo
    env[f'{name}_vo'] = monitor.vo
    env[f'{name}_k1'] = 4.3 * 40.3 * monitor.Eo * monitor.TE
    env[f'{name}_k2'] = 25 * monitor.Eo * monitor.TE
    env[f'{name}_k3'] = 1
%>
${name} = np.zeros((int(__nt / ${monitor.sample_interval}), __nr))
${name}_states = np.repeat([0., 1., 1., 1.], __nr).reshape((4, __nr))
${name}_i = 0
${name}_dt = __dt / 1000
</%def>

<%def name="bold_sample(name, monitor, env)">
${name}_s = ${name}_states[0] + ${name}_dt * (
    ${monitor.expression} - ${name}_itaus * ${name}_states[0] - ${name}_itauf * (${name}_states[1] - 1)
)
${name}_f = ${name}_states[1] + ${name}_dt * ${name}_states[0]
${name}_via = np.power(${name}_states[2], ${name}_ialpha)
${name}_v = ${name}_states[2] + ${name}_dt * (
    ${name}_itauo * (${name}_states[1] - ${name}_via)
)
${name}_eif = np.power(1 - ${name}_Eo, 1 / ${name}_states[1])
${name}_q = ${name}_states[3] + ${name}_dt * (
    ${name}_itauo * (
        ${name}_states[1] * (1 - ${name}_eif) / ${name}_Eo -
        ${name}_via * ${name}_states[3] / ${name}_states[2]
    )
)
${name}_states[0] = ${name}_s
${name}_states[1] = ${name}_f
${name}_states[2] = ${name}_v
${name}_states[3] = ${name}_q
if __it % ${monitor.sample_interval} == ${monitor.sample_interval} - 1:
    ${name}[${name}_i] = ${name}_vo * (
        ${name}_k1 * (1 - ${name}_q) +
        ${name}_k2 * (1 - ${name}_q / ${name}_v) + ${name}_k3 * (1 - ${name}_v)
    )
    ${name}_i += 1
</%def>