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
        
        par.R = 1.03
        par.beta = 1.0/par.R # Discount factor
        
        par.div_cost = 0.0
        par.div_A_share = 0.5 # divorce share of wealth to wife

        # Utility: gender-specific parameters
        par.rho_w = 2.0        # CRRA
        par.rho_m = 2.0        
        
        par.alpha_w = 1.0 # disutility weight of labor
        par.alpha_m = 1.0
        
        par.phi_w = 0.2 # curvature on disutility of labor
        par.phi_m = 0.2

        # wage process
        par.wage_const_w = 1.0
        par.wage_const_m = 1.0

        par.wage_K_w = 0.1
        par.wage_K_m = 0.1

        par.K_depre = 0.1
        
        # state variables
        par.T = 2
        
        # wealth
        par.num_A = 80
        par.max_A = 15.0

        # human capital
        par.num_K = 30
        par.max_K = 20.0

        par.sigma_K = 0.1
        par.num_shock_K = 5
        
        # bargaining power
        par.num_power = 21

        # love/match quality
        par.num_love = 10
        par.max_love = 1.0

        par.sigma_love = 0.1
        par.num_shock_love = 5 

        # simulation
        par.seed = 9210
        par.simT = par.T
        par.simN = 50_000

        # cpp
        par.threads = 16
        
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # setup grids
        self.setup_grids()
        
        # singles
        shape_single = (par.T,par.num_A,par.num_K)                        # single states: T and assets
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
        shape_couple = (par.T,par.num_power,par.num_love,par.num_A,par.num_K,par.num_K)     # states when couple: T, assets, power, love
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

        
        # simulation
        shape_sim = (par.simN,par.simT)
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

        # shocks
        np.random.seed(par.seed)
        sim.draw_love = np.random.normal(size=shape_sim)
        sim.draw_Kw = np.exp(-0.5*par.sigma_K**2 + par.sigma_K*np.random.normal(size=shape_sim))
        sim.draw_Km = np.exp(-0.5*par.sigma_K**2 + par.sigma_K*np.random.normal(size=shape_sim))

        # initial distribution
        sim.init_A = par.grid_A[0] + np.zeros(par.simN)
        sim.init_Aw = np.zeros(par.simN)
        sim.init_Am = np.zeros(par.simN)
        sim.init_couple = np.ones(par.simN,dtype=np.bool_)
        sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        sim.init_love = np.zeros(par.simN)
        sim.init_Kw = np.zeros(par.simN)
        sim.init_Km = np.zeros(par.simN)
        
        
    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A = nonlinspace(0.0,par.max_A,par.num_A,1.1)       # asset grid

        par.grid_Aw = par.div_A_share * par.grid_A                  # asset grid in case of divorce
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # power. non-linear grid with more mass in both tails.
        odd_num = np.mod(par.num_power,2)
        first_part = nonlinspace(0.0,0.5,(par.num_power+odd_num)//2,1.3)
        last_part = np.flip(1.0 - nonlinspace(0.0,0.5,(par.num_power-odd_num)//2 + 1,1.3))[1:]
        par.grid_power = np.append(first_part,last_part)

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
            par.grid_shock_K,par.grid_weight_K = np.array([0.0]),np.array([1.0])

        else:
            par.grid_shock_K,par.grid_weight_K = quadrature.log_normal_gauss_hermite(par.sigma_K,par.num_shock_K)


    def solve(self):
        
        sol = self.sol
        par = self.par 

        # setup grids
        self.setup_grids()

        self.cpp.solve(sol,par)


    def simulate(self):
        sol = self.sol
        sim = self.sim
        par = self.par

        self.cpp.simulate(sim,sol,par)

   