import numpy as np
import numba as nb
import scipy.optimize as optimize

from EconModel import EconModelClass
from consav.grids import nonlinspace
from consav import linear_interp, linear_interp_1d
from consav import quadrature

class LimitedCommitmentModelClass(EconModelClass):
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. cpp
        self.cpp_filename = 'cppfuncs/solve.cpp'
        self.cpp_options = {'compiler':'vs'}
        
    def setup(self):
        par = self.par
        
        par.R = 1.04
        #par.beta = 1.0/par.R # Discount factor
        par.beta = 0.98 # Discount factor
        
        par.div_cost = 0.0
        par.div_A_share = 0.5 # divorce share of wealth to wife

        # Utility: gender-specific parameters
        par.gamma1_w = 1.5        # CRRA  from Mariage labor supply and the Dynamics of social safety net
        par.gamma1_m = 1.5        # CRRA from Mariage labor supply and the Dynamics of social safety net

        
        par.gamma2_w = 0.1      
        par.gamma2_m = 0.1       

        
        par.gamma3_w = 1.5       
        par.gamma3_m= 1.5     


        # wage process
        par.kappa1 = 0.3 #proportionality of the tax system
        par.kappa2 = 0.185 #progression of the tax system from Heathcote et al
        #par.kappa2 = 0.0 #progression of the tax system from Heathcote et al


        par.wage_const_w = 0.01
        par.wage_const_m = 0.01

        par.wage_K_w = 0.5
        par.wage_K_m = 0.6
        par.wage_K_m = 0.5

        par.lambdaa2 = 1.0 #HK return to work              
        par.lambdaa2_HK = 0.2  #HK return to work  
        par.lambdaa3 = 0.03
        par.lambdaa4 = 0.0001
 
        par.K_depre = 0.1
        
        # state variables
        par.T = 2
        
        # wealth      
        par.num_A = 300 #300
        par.max_A = 10.0 # 30
        par.max_A = 7.0 # 30

        par.max_Aw = par.max_A*par.div_A_share 
        par.max_Am = par.max_A*(1-par.div_A_share )

        # human capital
        par.num_K = 10 #10
        par.max_K = 2.0 #1.0

        par.sigma_K = 0.2 #0.1
        par.num_shock_K = 7
        
        # bargaining power
        par.num_power = 11 #21

        # love/match quality
        par.num_love = 21 #81
        par.max_love = 1.0

        par.sigma_love = 0.031  # from Mariage labor supply and the Dynamics of social safety net
        par.num_shock_love = 7 

        #divorce quality
        par.pr_z = 0.05
        par.util_Z = -0.5

        #initial distribution factor
        par.pr_distr_factor = 0.5

        # simulation
        par.seed = 9210
        par.simN = 5_000

        # bargaining model
        par.bargaining = 1 # 0: no bargaining, full commitment, 1: limited commitment, 2: no commitment, 'Nash' bargaining
        par.bargaining_init_nash = False # True: Nash bargaining in initial period, else equal to par.bargaining in all periods
        
        # cpp
        par.threads = 16

        # post-decision states
        par.num_A_pd = 100 #100
        par.num_K_pd = 14


        #do Human capital
        par.do_HK = True


    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # setup grids
        self.setup_grids()
        
        # singles
        shape_single = (par.T, par.num_Z,par.num_A,par.num_K)                        # single states: T and assets
        sol.Vw_single = np.nan + np.ones(shape_single)
        sol.Vm_single = np.nan + np.ones(shape_single)      
        sol.labor_w_single = np.nan + np.ones(shape_single)     # labor supply, single women
        sol.labor_m_single = np.nan + np.ones(shape_single)     # labor supply, single men
        sol.cons_w_single = np.nan + np.ones(shape_single)     # consumption, single women
        sol.cons_m_single = np.nan + np.ones(shape_single)     # consumption, single men
        
        sol.Vw_trans_single = np.nan + np.ones(shape_single)
        sol.Vm_trans_single = np.nan + np.ones(shape_single)      
        sol.labor_w_trans_single = np.nan + np.ones(shape_single)     # labor supply, single women
        sol.labor_m_trans_single = np.nan + np.ones(shape_single)     # labor supply, single men
        sol.cons_w_trans_single = np.nan + np.ones(shape_single)     # consumption, single women
        sol.cons_m_trans_single = np.nan + np.ones(shape_single)     # consumption, single men
       
        # couples
        shape_couple = (par.T,par.num_Z, par.num_Z,par.num_power,par.num_love,par.num_A,par.num_K,par.num_K)     # states when couple: T, assets, power, love
        #shape_couple = (par.num_Z, par.num_Z)     # states when couple: T, assets, power, love
        
        sol.Vw_couple = np.nan + np.ones(shape_couple) # value of starting as couple
        sol.Vm_couple = np.nan + np.ones(shape_couple)
        sol.labor_w_couple = np.nan + np.ones(shape_couple)
        sol.labor_m_couple = np.nan + np.ones(shape_couple)
        sol.cons_w_couple = np.nan + np.ones(shape_couple)
        sol.cons_m_couple = np.nan + np.ones(shape_couple)
        
        sol.Vw_remain_couple = np.nan + np.ones(shape_couple)           # value marriage -> marriage
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple)
        sol.V_remain_couple = -np.inf + np.ones(shape_couple)                                                                       
        sol.labor_w_remain_couple = np.nan + np.ones(shape_couple)        
        sol.labor_m_remain_couple = np.nan + np.ones(shape_couple)        
        sol.cons_w_remain_couple = np.nan + np.ones(shape_couple)      
        sol.cons_m_remain_couple = np.nan + np.ones(shape_couple)      

        sol.power_idx = np.zeros(shape_couple,dtype=np.int_)            
        sol.power = np.zeros(shape_couple)   

        # post-decision pre-computation
        shape_pd = (par.num_Z, par.num_Z, par.num_power,par.num_love,par.num_A_pd,par.num_K_pd,par.num_K_pd)     
        sol.EVw_pd = np.nan + np.ones(shape_pd)
        sol.EVm_pd = np.nan + np.ones(shape_pd)                           

        
        # simulation
        shape_sim = (par.simN,par.T)
        sim.labor_w = np.nan + np.ones(shape_sim)               
        sim.labor_m = np.nan + np.ones(shape_sim)
        sim.cons_w = np.nan + np.ones(shape_sim)
        sim.cons_m = np.nan + np.ones(shape_sim)
        
        sim.A = np.nan + np.ones(shape_sim)
        sim.Aw = np.nan + np.ones(shape_sim)
        sim.Am = np.nan + np.ones(shape_sim)
        sim.couple = np.nan + np.ones(shape_sim)
        sim.power_idx = np.ones(shape_sim,dtype=np.int_)
        sim.power = np.nan + np.ones(shape_sim)
        sim.love = np.nan + np.ones(shape_sim)
        sim.Kw = np.nan + np.ones(shape_sim)
        sim.Km = np.nan + np.ones(shape_sim)
        
        sim.exp_w = np.nan + np.ones(shape_sim)
        sim.exp_m = np.nan + np.ones(shape_sim)
        sim.Zw = np.nan + np.ones(shape_sim)
        sim.Zm = np.nan + np.ones(shape_sim)
        sim.value = np.nan + np.ones(shape_sim)
        sim.util = np.nan + np.ones(shape_sim)

        # shocks
        np.random.seed(par.seed)
        sim.draw_love = par.sigma_love * np.random.normal(size=shape_sim)
        if par.do_HK:
            sim.draw_Kw_perm = np.exp(-0.5*par.sigma_K**2 + par.sigma_K*np.random.normal(size=shape_sim))
            sim.draw_Km_perm = np.exp(-0.5*par.sigma_K**2 + par.sigma_K*np.random.normal(size=shape_sim))
            sim.draw_Kw_temp = np.zeros(shape_sim)
            sim.draw_Km_temp = np.zeros(shape_sim)
        else:
            sim.draw_Kw_temp = par.sigma_K*np.random.normal(size=shape_sim)
            sim.draw_Km_temp = par.sigma_K*np.random.normal(size=shape_sim)
            sim.draw_Kw_perm = np.ones(shape_sim)
            sim.draw_Km_perm = np.ones(shape_sim)
            
        sim.draw_Zw = np.random.uniform(size=shape_sim)
        sim.draw_Zm = np.random.uniform(size=shape_sim) 

        # initial distribution
        sim.init_A = par.grid_A[0] + np.zeros(par.simN)
        sim.init_Aw = par.grid_Aw[0] + np.zeros(par.simN)
        sim.init_Am = par.grid_Am[0] +  np.zeros(par.simN)
        sim.init_couple = np.ones(par.simN,dtype=np.bool_)
        sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        sim.init_love = np.zeros(par.simN)
        
        sim.init_Kw = np.random.uniform(low=0.0, high =0.2,size=shape_sim)
        sim.init_Km = np.random.uniform(low=0.0, high =0.2,size=shape_sim)
        #sim.init_Kw = np.exp(-0.5*par.sigma_K_init**2 + par.sigma_K_init*np.random.normal(size=par.simN))
        #sim.init_Km = np.exp(-0.5*par.sigma_K_init**2 + par.sigma_K_init*np.random.normal(size=par.simN))
        sim.init_Zw = np.zeros(par.simN)
        sim.init_Zm = np.zeros(par.simN)
        #sim.init_Zw = np.random.choice(2,par.simN)
        #sim.init_Zm = np.random.choice(2,par.simN)


        sim.init_distr = np.random.choice(3,par.simN, p =[par.pr_distr_factor*(1-par.pr_distr_factor),par.pr_distr_factor*par.pr_distr_factor+(1-par.pr_distr_factor)*(1-par.pr_distr_factor),par.pr_distr_factor*(1-par.pr_distr_factor)])
        sim.init_distr_power_lag = np.random.uniform(low=0.0, high =0.4, size=par.simN)
        
        
    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A = nonlinspace(1.0e-2,par.max_A,par.num_A,1.5)       # asset grid

        par.grid_Aw = par.div_A_share * par.grid_A                  # asset grid in case of divorce
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # power. non-linear grid with more mass in both tails.
        odd_num = np.mod(par.num_power,2)
        first_part = nonlinspace(0.0,0.5,(par.num_power+odd_num)//2,1.3)
        last_part = np.flip(1.0 - nonlinspace(0.0,0.5,(par.num_power-odd_num)//2 + 1,1.3))[1:]
        par.grid_power = np.append(first_part,last_part)
        par.grid_power_flip = np.flip(par.grid_power) # flip for men

        # love grid and shock
        if par.num_love>1:
            par.grid_love = np.linspace(-par.max_love,par.max_love,par.num_love)
        else:
            par.grid_love = np.array([0.0])

        if par.sigma_love<=1.0e-6:
            par.num_shock_love = 1
            par.grid_shock_love,par.grid_weight_love = np.array([0.0]),np.array([1.0])

        else:
            par.grid_shock_love,par.grid_weight_love = quadrature.normal_gauss_hermite(par.sigma_love,par.num_shock_love)

        # human capital
        par.grid_K = nonlinspace(0.0,par.max_K,par.num_K,1.1)   

        if par.sigma_K<=1.0e-6:
            par.num_shock_K = 1
            par.grid_shock_K_perm,par.grid_weight_K = np.array([0.0]),np.array([1.0])

        else:
            if par.do_HK:
                #if do HK, permanent wage shock
                par.grid_shock_K_perm,par.grid_weight_K = quadrature.log_normal_gauss_hermite(par.sigma_K,par.num_shock_K)
                par.grid_shock_K_temp = np.zeros(par.num_shock_K)
            else:
                #if not HK temporary wage shock
                par.grid_shock_K_temp,par.grid_weight_K = quadrature.normal_gauss_hermite(par.sigma_K,par.num_shock_K)
                par.grid_shock_K_perm = np.ones(par.num_shock_K)


        #divorce utility grid
        par.num_Z = 2 #high or low bmi
        par.grid_Z = np.array([0.0,1.0])
        par.grid_weight_Z = np.array([[1-par.pr_z,par.pr_z],[par.pr_z, 1-par.pr_z]])

        #initially distrbution factor
        par.num_distr_factor = 3 #high or low bmi
        par.grid_distr_factor = np.array([0,1,2]) #woman highest, equal, woman lowest
        #par.pr_distr_factor = np.array([[par.pr_distr_factor*(1-par.pr_distr_factor),par.pr_distr_factor*par.pr_distr_factor+(1-par.pr_distr_factor)*(1-par.pr_distr_factor),par.pr_distr_factor*(1-par.pr_distr_factor)]])

        # post-decision states
        par.grid_A_pd = nonlinspace(1.0e-2,par.max_A,par.num_A_pd,1.1)       # asset grid
        if par.do_HK: 
            par.grid_K_pd = nonlinspace(0.0,par.max_K,par.num_K_pd,1.1)       # human capital grid
        else: 
            par.num_K_pd = 1
            par.grid_K_pd = np.array([1.0])


    def solve(self):
        
        sol = self.sol
        par = self.par 

        # setup grids
        # self.setup_grids() #allocate? would be a bit slower but more safe
        self.allocate()

        self.cpp.solve(sol,par)


    def simulate(self):
        sol = self.sol
        sim = self.sim
        par = self.par

        self.cpp.simulate(sim,sol,par)

   