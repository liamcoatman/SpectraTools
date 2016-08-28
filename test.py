from lmfit import Parameters, Model

def fit_model(x, p):

	return p['a'].value + p['b'].value + p['c'].value 

pars = Parameters()

pars.add('a', value = 1.0)
pars.add('b', value = 2.0)
pars.add('c', value = 3.0)

mod = Model(fit_model)

print mod.eval(p=pars, x=5.0)


