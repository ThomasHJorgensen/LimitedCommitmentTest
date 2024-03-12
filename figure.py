#FIGURE
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt


# plot style
linestyles = ['-','--','-.',':',':']
markers = ['o','s','D','*','P']
linewidth = 2
font_size = 17
font = {'size':font_size}
matplotlib.rc('font', **font)
plt.rcParams.update({'figure.max_open_warning': 0,'text.usetex': False})


def figure_single(model, var_list, t_list,  i_A , i_Z , i_HK , model_name = 'model', path = 'output/'):
    alpha = 0.5 
    cmaps = ('viridis','gray')
    for t in t_list:
        for var in var_list:
            sol = model.sol
            par = model.par

            var_now = f'{var}_single'

            
            var_name = f'{model_name}_single_{var}'

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X, Y = np.meshgrid(par.grid_Aw, par.grid_K,indexing='ij')                   
            Z = getattr(model.sol,var_now)[t,0]
            ax.plot_surface(X, Y,Z,cstride=1,cmap=cmaps[0], edgecolor='none',alpha=alpha)       
            ax.set(xlabel='$A_{j,t-1}$',ylabel='$K_{j,t}$', zlabel=var)
            plt.savefig(f'{path}{var_name}_gridA_gridK_t{t}.png')
   
            plt.close(fig)

            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            Z1 = getattr(model.sol,var_now)[t,0,:,i_HK]
            Z2 = getattr(model.sol,var_now)[t,1,:,i_HK]
            ax.plot(par.grid_A,Z1, label = 'single Z=0')
            ax.plot(par.grid_A,Z2, label = 'single Z=1')                    
            ax.set_xlabel(f"$A$")
            ax.set_ylabel(var)       
            plt.savefig(f'{path}{var_name}_gridA_t{t}.png')
            plt.close(fig)

            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            Z1 = getattr(model.sol,var_now)[t,0,i_A,:]
            Z2 = getattr(model.sol,var_now)[t,1,i_A,:]
            ax.plot(par.grid_K,Z1, label = 'single woman Z=0')
            ax.plot(par.grid_K,Z2, label = 'single woman Z=1')                 
            ax.set_xlabel(f"$K$")
            ax.set_ylabel(var)     
            plt.savefig(f'{path}{var_name}_{var}_gridK_t{t}.png')
            plt.legend()
            plt.close(fig)


def figure_couple(model, var_list, t_list, i_A , i_Zw , i_Zm, i_HKw , i_HKm, i_L  , i_P , sol_type = 'couple',  model_name = 'model', path = 'output/'):
   
    alpha = 0.5 

    cmaps = ('viridis','gray')
    
    for t in t_list:
        for var in var_list:

                    
            sol = model.sol
            par = model.par

            var_now = f'{var}_{sol_type}'
            var_name = f'{model_name}_{sol_type}_{var}'

            
            
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X, Y = np.meshgrid(par.grid_A, par.grid_K,indexing='ij') 
            Z = getattr(model.sol,var_now)[t,0,0,i_P,i_L,:,:,i_HKm]
            ax.plot_surface(X, Y,Z,cstride=1,cmap=cmaps[0], edgecolor='none',alpha=alpha);       
            ax.set(xlabel='$A_{j,t-1}$',ylabel='$K_{w,t}$', zlabel=var);
            plt.savefig(f'{path}{var_name}gridA_gridKw_t{t}.png')
            plt.close(fig)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X, Y = np.meshgrid(par.grid_K, par.grid_K,indexing='ij')
            Z = getattr(model.sol,var_now)[t,0,0,i_P,i_L,i_A,:,:]
            ax.plot_surface(X, Y,Z,cstride=1,cmap=cmaps[0], edgecolor='none',alpha=alpha);       
            ax.set(xlabel='$K_{w,t}$',ylabel='$K_{m,t}$', zlabel=var);
            plt.savefig(f'{path}{var_name}gridAKw_gridKm_t{t}.png')
            plt.close(fig)




            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X, Y = np.meshgrid(par.grid_power, par.grid_A,indexing='ij')     
            Z = getattr(model.sol,var_now)[t,0,0,:,i_L,:,i_HKw,i_HKm]   
            ax.plot_surface(X, Y,Z,cstride=1,cmap=cmaps[0], edgecolor='none',alpha=alpha);       
            ax.set(xlabel='$P_{t}$',ylabel='$A_{j,t-1}$', zlabel=var);
            plt.savefig(f'{path}{var_name}gridP_gridA_t{t}.png')
            plt.close(fig)



            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X, Y = np.meshgrid(par.grid_love, par.grid_A,indexing='ij')     
            Z = getattr(model.sol,var_now)[t,0,0,i_P,:,:,i_HKw,i_HKm]
            ax.plot_surface(X, Y,Z,cstride=1,cmap=cmaps[0], edgecolor='none',alpha=alpha);       
            ax.set(xlabel='$L_{t}$',ylabel='$A_{j,t-1}$', zlabel=var);
            plt.savefig(f'{path}{var_name}gridL_gridA_t{t}.png')
            plt.close(fig)




            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            Z1 = getattr(model.sol,var_now)[t,0,0,i_P,i_L,:,i_HKw,i_HKm]
            Z2 = getattr(model.sol,var_now)[t,0,1,i_P,i_L,:,i_HKw,i_HKm]
            Z3 = getattr(model.sol,var_now)[t,1,0,i_P,i_L,:,i_HKw,i_HKm]
            Z4 = getattr(model.sol,var_now)[t,1,1,i_P,i_L,:,i_HKw,i_HKm]
            ax.plot(par.grid_A,Z1, label = '{sol_type} woman Zw=0, Zm=0')
            ax.plot(par.grid_A,Z2, label = '{sol_type}  woman Zw=0, Zm=1')
            ax.plot(par.grid_A,Z3, label = '{sol_type}  woman Zw=1, Zm=0')
            ax.plot(par.grid_A,Z4,label = '{sol_type}  woman Zw=1, Zm=1')                      
            ax.set_xlabel(f"$A$")
            ax.set_ylabel(var)
            plt.savefig(f'{path}{var_name}gridA_t{t}.png')
            plt.close(fig)



            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            Z1 = getattr(model.sol,var_now)[t,0,0,i_P,i_L,i_A,:,i_HKm]
            Z2 = getattr(model.sol,var_now)[t,0,1,i_P,i_L,i_A,:,i_HKm]
            Z3 = getattr(model.sol,var_now)[t,1,0,i_P,i_L,i_A,:,i_HKm]
            Z4 = getattr(model.sol,var_now)[t,1,1,i_P,i_L,i_A,:,i_HKm]
            ax.plot(par.grid_K,Z1, label = '{sol_type} woman Zw=0, Zm=0')
            ax.plot(par.grid_K,Z2, label = '{sol_type} woman Zw=0, Zm=1')
            ax.plot(par.grid_K,Z3, label = '{sol_type} woman Zw=1, Zm=0')
            ax.plot(par.grid_K,Z4, label = '{sol_type} woman Zw=1, Zm=1')                      
            ax.set_xlabel(f"$Kw$")
            ax.set_ylabel(var)
            plt.savefig(f'{path}{var_name}gridK_t{t}.png')
            plt.close(fig)


            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            Z1 = getattr(model.sol,var_now)[t,0,0,i_P,:,i_A,i_HKw,i_HKm]
            Z2 = getattr(model.sol,var_now)[t,0,1,i_P,:,i_A,i_HKw,i_HKm]
            Z3 = getattr(model.sol,var_now)[t,1,0,i_P,:,i_A,i_HKw,i_HKm]
            Z4 = getattr(model.sol,var_now)[t,1,1,i_P,:,i_A,i_HKw,i_HKm]
            ax.plot(par.grid_love,Z1, label = '{sol_type} woman Zw=0, Zm=0')
            ax.plot(par.grid_love,Z2, label = '{sol_type} woman Zw=0, Zm=1')
            ax.plot(par.grid_love,Z3, label = '{sol_type} woman Zw=1, Zm=0')
            ax.plot(par.grid_love,Z4, label = '{sol_type} woman Zw=1, Zm=1')                           
            ax.set_xlabel(f"$Love$")
            ax.set_ylabel(var)
            plt.savefig(f'{path}{var_name}gridL_t{t}.png')
            plt.close(fig)


            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)
            Z1 = getattr(model.sol,var_now)[t,0,0,:,i_L,i_A,i_HKw,i_HKm]
            Z2 = getattr(model.sol,var_now)[t,0,1,:,i_L,i_A,i_HKw,i_HKm]
            Z3 = getattr(model.sol,var_now)[t,1,0,:,i_L,i_A,i_HKw,i_HKm]
            Z4 = getattr(model.sol,var_now)[t,1,1,:,i_L,i_A,i_HKw,i_HKm]
            ax.plot(par.grid_power,Z1, label = '{sol_type} woman Zw=0, Zm=0')
            ax.plot(par.grid_power,Z2, label = '{sol_type} woman Zw=0, Zm=1')
            ax.plot(par.grid_power,Z3, label = '{sol_type} woman Zw=1, Zm=0')
            ax.plot(par.grid_power,Z4, label = '{sol_type} woman Zw=1, Zm=1')                           
            ax.set_xlabel(f"$Power$")
            ax.set_ylabel(var)
            plt.legend()
            plt.savefig(f'{path}{var_name}gridP_t{t}.png')
            plt.close(fig)






