import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import Estimate
from EconModel import cpptools

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
threads = 50

#cpptools.setup_nlopt(folder='cppfuncs/', do_print = True) #install nlopt
# problem: Mange perioder, værdien af HK i første periode er meget lavt, hvis vi ikke har HK, måske ændre til type? Type med højt return og type med lavt return


specs = {}
T = 5
sigma_love = 0.031
sigma_HK = 0.1
sigma_HK_init = 0.1
#specs.update({f'model_FC_NO_HK_T{T}_sHK{sigma_HK}_initHK{sigma_HK_init}_SL{sigma_love}':{'latexname':'limited', 'par':{ 'T':T, 'sigma_K':sigma_HK, 'sigma_K_init': sigma_HK_init, 'sigma_love': sigma_love, 'do_HK': False, 'threads':threads,'bargaining':0}}})
#specs.update({f'model_LC_NO_HK_T{T}_sHK{sigma_HK}_initHK{sigma_HK_init}_SL{sigma_love}':{'latexname':'limited', 'par':{ 'T':T, 'sigma_K':sigma_HK, 'sigma_K_init': sigma_HK_init, 'sigma_love': sigma_love, 'do_HK': False, 'threads':threads,'bargaining':1}}})
#specs.update({f'model_NC_NO_HK_T{T}_sHK{sigma_HK}_initHK{sigma_HK_init}_SL{sigma_love}':{'latexname':'limited', 'par':{ 'T':T, 'sigma_K':sigma_HK, 'sigma_K_init': sigma_HK_init, 'sigma_love': sigma_love, 'do_HK': False, 'threads':threads,'bargaining':2}}})

specs.update({f'model_NC_add_shock':{'latexname':'limited', 'par':{ 'T':T, 'sigma_K':sigma_HK, 'sigma_K_init': sigma_HK_init, 'sigma_love': sigma_love, 'do_HK': False, 'threads':threads,'bargaining':2}}})



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
    
    #Save the data 
    T = models[name].par.T
    print(' saving...')
    models[name].sim.init_love[:] = 0.0
    models[name].sim.init_A[:] = 0.0
    np.random.seed(models[name].par.seed)
    data1 = Estimate.create_data_new(models[name],start_p = 1, end_p = T-1, to_xl = True, name_xl = name)

    print(f'Couple {np.mean(models[name].sim.couple,0)}')
    
    print(f'HKw {np.mean(models[name].sim.Kw,0)}')
    print(f'HKm {np.mean(models[name].sim.Km,0)}')
    print(f'Laborw {np.mean(models[name].sim.labor_w,0)}')
    print(f'Laborm {np.mean(models[name].sim.labor_m,0)}')
    print(f'Asset {np.nanmean(models[name].sim.A,0)}')



#model = models['model FC, NO_HK ']
#model.sim.init_love[:] =0.2
##model.simulate()
#print('HK')*
#print(f'Couple {np.mean(model.sim.couple,0)}')

#print(f'Laborw {np.mean(model.sim.labor_w,0)}')
#print(f'Laborm {np.mean(model.sim.labor_m,0)}')
#print(f'Consumption W {np.mean(model.sim.cons_w,0)}')
#print(f'Consumption M {np.mean(model.sim.cons_m,0)}')
#print(f'Asset {np.nanmean(model.sim.A,0)}')
#print(f'HKw {np.mean(model.sim.Kw,0)}')
#print(f'HKm {np.mean(model.sim.Km,0)}')
#temp = model.sim.power
#I = model.sim.couple<1
#nan = np.zeros(I.shape)
#nan[I] = np.nan
#temp = np.nanmean(temp + nan,axis=0)
#print(f'Power {temp}')
