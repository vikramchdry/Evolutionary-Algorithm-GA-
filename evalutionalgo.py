import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.animation

DNA_SIZE = 1  
DNA_BOUND = [0, 5]
N_GENERATIONS = 300
MUT_STRENGTH = 10

def F(x):
	return np.sin(10*x)*x + np.cos(2*x)*x    # to find maximux for this function



def fit(pred):
	return pred.flatten()



def make_kid(parent):
	k = parent + MUT_STRENGTH*np.random.random(DNA_SIZE)
	k = np.clip(k, *DNA_BOUND)
	return k



def kill_bad(parent,kid):
	global MUT_STRENGTH
	fp = fit(F(parent))[0]
	fk = fit(F(kid))[0]
	p_target = 1/5
	if fp <fk:
		parent = kid
		ps = 1

	else:
		ps = 0 
	MUT_STRENGTH =  np.exp(1/np.sqrt(DNA_SIZE+1)*(ps-p_target)/(1-p_target))
	return parent


parent = 5 * np.random.rand(DNA_SIZE)
plt.ion() 
x = np.linspace(*DNA_BOUND, 200)

for generation in range(N_GENERATIONS):

	kid = make_kid(parent)
	py, ky = F(parent), F(kid) 
	parent = kill_bad(parent, kid)
	#print("Num of Generation :", N_GENERATIONS,'| MUT_STRENGTH: %.2f' % MUT_STRENGTH)



	plt.cla()
	plt.scatter(parent, py, s=200, lw=0, c='red', alpha=0.5,)
	plt.scatter(kid, ky, s=200, lw=0, c='blue', alpha=0.5)
	plt.text(0, -7, 'Mutation strength=%.2f' % MUT_STRENGTH)
	plt.plot(x, F(x)); plt.pause(0.05)



plt.ioff();
plt.show()
