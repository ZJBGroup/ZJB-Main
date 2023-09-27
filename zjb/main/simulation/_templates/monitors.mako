<%def name="raw_init(name, monitor, env)">
${name} = np.zeros((__nt, __nr))
</%def>

<%def name="raw_sample(name, monitor, env)">
${name}[__it] = ${monitor.expression}
</%def>
