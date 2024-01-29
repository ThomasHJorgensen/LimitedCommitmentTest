#test


import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import Estimate

from LimitedCommitmentModel import LimitedCommitmentModelClass

# plot style
linestyles = ['-','--','-.',':',':']
markers = ['o','s','D','*','P']
linewidth = 2
font_size = 17
font = {'size':font_size}
matplotlib.rc('font', **font)


plt.rcParams.update({'figure.max_open_warning': 0,'text.usetex': False})
path = 'output/'

SAVE = False

# c++ settings
do_compile = True
threads = 7

# compile c++ files
T = 5
specs = {  
    #'model 1':{'latexname':'limited', 'par':{ 'T':T,'threads':threads,'bargaining':0}},
    'model 2':{'latexname':'limited', 'par':{ 'T':T, 'do_HK': False, 'threads':threads,'bargaining':1}},
    #'model 3':{'latexname':'limited', 'par':{ 'T':T,'threads':threads,'bargaining':2}},
}




# solve different models
models = {}
for m,(name,spec) in enumerate(specs.items()):
    print(f'{name} loading...',end='')
    
    # setup model
    models[name] = LimitedCommitmentModelClass(name=name,par=spec['par'])
    models[name].spec = spec

    compile_now = True if do_compile & (m==0) else False
    models[name].link_to_cpp(force_compile=compile_now)
    
    print(' solving...')
    models[name].solve() 