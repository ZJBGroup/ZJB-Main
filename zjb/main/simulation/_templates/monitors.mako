<%def name="raw_init(name, monitor)">
${name} = np.zeros((__nt, __nr))
</%def>

<%def name="raw_sample(name, monitor)">
${name}[__it] = ${monitor.expression}
</%def>
