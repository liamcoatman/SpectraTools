# only confident this works for gaussians 

import numpy as np 
l = [] 
with open('/data/lc585/WHT_20150331/OBS/TS1049+1032_LR_Night2/Reduced/splot.log') as f:
    for line in f.readlines():
        if len(line.split()) == 7:
        	l.append(line.split())

l = np.asarray(l, dtype=np.float)

# in mk1dspec peak normalised to continum of one.
# use gaussian - voigt seems to overestimate bkgd for some reason 
# wavelength peak gaussian gfwhm    
with open('/data/lc585/WHT_20150331/OBS/TS1049+1032_LR_Night2/Reduced/splot_mod.log', 'w') as f:
    for line in l:
    	f.write(str(line[0]) + ' ' + str(1.0-line[1]+line[4]) + ' gaussian ' + str(line[5]) + '\n')