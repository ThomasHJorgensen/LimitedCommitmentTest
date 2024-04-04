
#import
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import statsmodels.api as sm

def create_data(model,start_p = 1, end_p = 4, to_xl = False, name_xl = 'simulated_data',path='output/'):
    
    par = model.par 
    sim = model.sim 

    #update stockastic elements.
    shape_sim = (par.simN,par.T)
    sim.draw_love = par.sigma_love * np.random.normal(size=shape_sim)
    sim.draw_Kw = par.sigma_K * np.random.normal(size=shape_sim)
    sim.draw_Km = par.sigma_K * np.random.normal(size=shape_sim)

    #simulate the data
    model.simulate()

    #Save the data in a data frame
    data_nu = {}
    data   = pd.DataFrame()

    #TODO: save it while simulating
    wage_w      =  np.exp(model.par.wage_const_w +model.par.wage_K_w*model.sim.Kw)  
    y_w         =  wage_w*model.sim.labor_w
    wage_m      =  np.exp(model.par.wage_const_m+model.par.wage_K_m* model.sim.Km)
    y_m         =  wage_m*model.sim.labor_m
    init_barg   =  model.sim.init_distr
    Z_w         =  model.sim.Zw 
    Z_m         =  model.sim.Zm 
    for i in range(start_p, end_p): #use some periods in the middle of the simluation

        # WOMAN
        #value this period and last period, 
        data_nu[i] = pd.DataFrame({
            'idx': range(1,model.par.simN+1) ,
            't' : i,
            'wealth' : model.sim.A[:,i] ,
            'couple': model.sim.couple[:,i],
            'hours_w': model.sim.labor_w[:,i],
            'hours_m': model.sim.labor_m[:,i],
            'cons': model.sim.cons_w[:,i],
            'wage_w': wage_w[:,i],
            'wage_m': wage_m[:,i],
            'earnings_w': y_w[:,i],
            'earnings_m': y_m[:,i],
            'BMI_w': Z_w[:,i],
            'BMI_m': Z_m[:,i],
            'omega_w':  model.sim.draw_Kw[:,i],
            'omega_m':  model.sim.draw_Km[:,i],
            'init_barg': init_barg,
            'value'    : model.sim.value[:,i],
            'util' : model.sim.util[:,i],
            'barganing': model.sim.power[:,i],
            'Love': model.sim.love[:,i],
            'exp_w': model.sim.exp_w[:,i],
            'exp_m': model.sim.exp_m[:,i]
        })

        

        #collect the data
        data = pd.concat([data,data_nu[i]] )


    


    #sort data
    data = data.sort_values(by =['idx','t'])
    if to_xl: 
        data.to_excel(f'{path}{name_xl}.xlsx')

    #create variable
    #data = create_variable(data)


    return data

def create_variable(data, par, print_aux_reg = False):
    data = data.sort_values(by =['idx','t'])

    data = data.drop(columns = ['Love','util','value','exp_w','exp_m','barganing'])


    #change BMI to a 1-2 instead of 0 -1 (so I can take the logarithm)
    data['BMI_w'] = 1 + data['BMI_w'] 
    data['BMI_m'] = 1 + data['BMI_m'] 



    ## change the period for wealth (from end og period to begining period)
    #data['wealth_temp'] = data['wealth'].shift(-1)
    #data['wealth_temp'] = np.where(data['t']== models.par.T-2, np.nan,data['wealth_temp'])

        
    #drop if single
    data = data[data['couple']>0.5]
    data = data[data['t']>8]
    data = data[data['t']<15]


    data = data.drop(columns = ['couple'])

    #drop if not paricipating
    data = data[data['hours_w']>0.001]
    data = data[data['hours_m']>0.001]

    #gen faminc and income share
    data['fam_inc'] = data['earnings_w']+data['earnings_m']
    data['inc_share_w'] = data['earnings_w']/data['fam_inc']
    data['inc_share_m'] = data['earnings_m']/data['fam_inc']
    
    #create dummy 
    #data['tt'] = data['t']
    #data=pd.get_dummies(data, columns = ['tt'], prefix = 'D_t', dtype = float) 
    #data=pd.get_dummies(data, columns = ['init_barg'], prefix = 'D_init_barg', dtype = float) 


    #transform with  lag 
    list = ['t','idx','BMI_w', 'BMI_m', 'omega_w', 'omega_m', 'hours_w','hours_m', 'cons','wage_w','wage_m','earnings_w','earnings_m','fam_inc','wealth','inc_share_w','inc_share_m']

    data_l = data.loc[:,list]
    data_l['t'] = data_l['t']+1

    data_l2 = data.loc[:,list]
    data_l2['t'] = data_l2['t']+2

    data_l3 = data.loc[:,list]
    data_l3['t'] = data_l3['t']+3

    
    data_F = data.loc[:,['t','idx','wealth']]
    data_F['t'] = data_F['t']-1

    #transfor with logs
    list_log = ['BMI_w', 'BMI_m', 'hours_w','hours_m', 'cons','wage_w','wage_m','earnings_w','earnings_m','fam_inc','wealth']
    #transfor with logs
    for val in list_log:
        # take the log
        log_name = 'log_' + val
        data[log_name] = np.log(data[val])
        data_l[log_name] = np.log(data_l[val])
        data_l2[log_name] = np.log(data_l2[val])
        data_l3[log_name] = np.log(data_l3[val])


    data =  data.merge(data_l, how='left', left_on = ['t','idx'], right_on = ['t','idx'], suffixes=('', '_l'))
    data =  data.merge(data_l2, how='left', left_on = ['t','idx'], right_on = ['t','idx'], suffixes=('', '_l2'))
    data =  data.merge(data_l3, how='left', left_on  = ['t','idx'], right_on = ['t','idx'], suffixes=('', '_l3'))
    data =  data.merge(data_F, how='left', left_on  = ['t','idx'], right_on = ['t','idx'], suffixes=('', '_F'))

    #find the differencce
    for val in list_log:
        #delta
        delta_log       = 'delta_log_' + val
        delta_log_l     = 'delta_log_' + val + '_l'
        delta_log_l2    = 'delta_log_' + val + '_l2'
        
        log_name       = 'log_' + val
        log_name_l     = 'log_' + val + '_l'
        log_name_l2     = 'log_' + val + '_l2'
        log_name_l3     = 'log_' + val + '_l3'
        data[delta_log] = data[log_name]-data[log_name_l]
        data[delta_log_l] = data[log_name_l]-data[log_name_l2]
        data[delta_log_l2] = data[log_name_l2]-data[log_name_l3]


    #DROP NAN
    data = data.dropna(subset = ['delta_log_wage_w','delta_log_wage_m'])

    
    #ESTIMATE OMEGA
    data = aux_est(data, par, print_reg = print_aux_reg)
    list = ['t','idx','omega_res_w', 'omega_res_m']
        
    data_l = data.loc[:,list]
    data_l['t'] = data_l['t']+1
        
    data_l2 = data.loc[:,list]
    data_l2['t'] = data_l2['t']+2
    data_l3 = data.loc[:,list]
    data_l3['t'] = data_l3['t']+3

        
    data =  data.merge(data_l, how='left', left_on = ['t','idx'], right_on = ['t','idx'], suffixes=('', '_l'))
    data =  data.merge(data_l2, how='left', left_on = ['t','idx'], right_on = ['t','idx'], suffixes=('', '_l2'))
    data =  data.merge(data_l3, how='left', left_on = ['t','idx'], right_on = ['t','idx'], suffixes=('', '_l3'))
 

    
    #vairable for regression
    #X1 = data['omega_w']-data['omega_w_l']
    #X2 = data['omega_m']-data['omega_m_l']
    #X3 = data['delta_log_hours_w']*1/data['hours_w_l']
    #X4 = data['delta_log_hours_m']*1/data['hours_m_l']
    #X5 = data['inc_share_m']*data['delta_log_earnings_m']
    #X6 = data['inc_share_w']*data['delta_log_earnings_w']
    #X7 = data['cons_l']*data['delta_log_cons']

    
    #data = pd.concat([data, X1.rename('delta_omega_w'),X2.rename('delta_omega_m'),X3.rename('y_w'), X4.rename('y_m'), X5.rename('control_part_inc_w '), X6.rename('control_part_inc_m'), X7.rename('control_cons')],  axis = 1 ,ignore_index = True) #in issue from middle of marts, try to check later in april to see wheter it is still an issue
    data['delta_omega_w'] = data['omega_w']-data['omega_w_l']
    data['delta_omega_m'] = data['omega_m']-data['omega_m_l']    
    data['delta_omega_w_l'] = data['omega_w_l']-data['omega_w_l2']
    data['delta_omega_m_l'] = data['omega_m_l']-data['omega_m_l2']    
    data['delta_omega_w_l2'] = data['omega_w_l2']-data['omega_w_l3']
    data['delta_omega_m_l2'] = data['omega_m_l2']-data['omega_m_l3']
    data['y_w'] = data['delta_log_hours_w']*data['hours_w_l']
    data['y_m'] = data['delta_log_hours_m']*data['hours_m_l']
    data['control_part_inc_w'] = data['inc_share_m_l']*data['delta_log_earnings_m']
    data['control_part_inc_m'] = data['inc_share_w_l']*data['delta_log_earnings_w']
    data['control_cons'] = data['cons_l']*data['delta_log_cons']

    
    data = data[['t','idx','init_barg', 'inc_share_w', 'inc_share_w_l', 'delta_log_wage_w','delta_log_wage_m','delta_log_fam_inc', 'omega_res_w', 'omega_res_m','omega_res_w_l', 'omega_res_m_l','omega_res_w_l2', 'omega_res_m_l2', 'delta_omega_m','delta_omega_w','delta_omega_w_l', 'delta_omega_m_l','delta_omega_w_l2', 'delta_omega_m_l2', 'y_w', 'y_m', 'control_part_inc_w', 'control_part_inc_m', 'control_cons', 'delta_log_wealth','delta_log_wealth_l', 'delta_log_wealth_l2' ,'delta_log_fam_inc', 'log_fam_inc', 'log_wealth', 'log_fam_inc_l', 'log_wealth_l', 'wealth_F', 'log_earnings_w', 'log_earnings_m', 'log_earnings_w_l', 'log_earnings_m_l', 'delta_log_BMI_w', 'delta_log_BMI_m', 'delta_log_BMI_w_l', 'delta_log_BMI_m_l', 'delta_log_BMI_w_l2', 'delta_log_BMI_m_l2']]
    return data

def aux_est(data, par, print_reg = False):

    
    #estimate the wage shock
    x=pd.get_dummies(data['t'], columns = ['t'], prefix = 'D_t', dtype = float) 
    y  = data['delta_log_wage_w']
    x = sm.add_constant(x) 

    result = sm.OLS(y,x).fit()
    if print_reg:
        print('Residuals from own wage equation, w')
        print(result.summary())
    data['omega_res_w'] = result.resid 


    y  = data['delta_log_wage_m']
    result = sm.OLS(y,x).fit()
    if print_reg:
        print('Residuals from own wage equation, w')
        print(result.summary())
    data['omega_res_m'] = result.resid 

    return data


def main_est(data, gender = "w", do_estimate_omega = True, print_reg = False, shadow_value_simple = True):
    #gender = w, woman, 

    if gender == "w":
        spouse = "m"
    else:
        spouse = "w"
        gender = "m"

    #FOR WOMAN
    if do_estimate_omega:
        data['wage_shock']=data[f'omega_res_{gender}']
        data['wage_shock_l']=data[f'omega_res_{gender}_l']
        data['wage_shock_l2']=data[f'omega_res_{gender}_l2']
        data['wage_shock_j']=data[f'omega_res_{spouse}_m']
        data['wage_shock_j_l']=data[f'omega_res_{spouse}_l']
        data['wage_shock_j_l2']=data[f'omega_res_{spouse}_l2']
    else:
        data['wage_shock']=data[f'delta_omega_{gender}']
        data['wage_shock_l']=data[f'delta_omega_{gender}_l']
        data['wage_shock_l2']=data[f'delta_omega_{gender}_l2']
        data['wage_shock_j']=data[f'delta_omega_{spouse}']
        data['wage_shock_j_l']=data[f'delta_omega_{spouse}_l']
        data['wage_shock_j_l2']=data[f'delta_omega_{spouse}_l2']

    data['BMI'] = data[f'delta_log_BMI_{gender}']
    data['BMI_l'] = data[f'delta_log_BMI_{gender}_l']
    data['BMI_l2'] = data[f'delta_log_BMI_{gender}_l2']
    data['BMI_j'] = data[f'delta_log_BMI_{spouse}']
    data['BMI_j_l'] = data[f'delta_log_BMI_{spouse}_l']
    data['BMI_j_l2'] = data[f'delta_log_BMI_{spouse}_l2']

    
    
    
    data_regress = data[['t','init_barg','log_earnings_w', 'log_earnings_m','log_earnings_w_l', 'log_earnings_m_l', 'inc_share_w', 'inc_share_w_l','log_wealth', 'wealth_F',  f'y_{gender}', 'idx', 'wage_shock','wage_shock_l','wage_shock_l2','wage_shock_j','wage_shock_j_l','wage_shock_j_l2','BMI','BMI_l','BMI_l2','BMI_j','BMI_j_l','BMI_j_l2',f'control_part_inc_{gender}','control_cons','delta_log_wealth','delta_log_wealth_l','delta_log_wealth_l2','delta_log_fam_inc', 'log_fam_inc_l', 'log_wealth_l']]

    #DROP NAN
    data_regress = data_regress.dropna() # det ser ud som om den ikke fjerner nogen

    #PREPARE T
    X_t=pd.get_dummies(data_regress[['t', 'init_barg']], columns = ['t','init_barg'], prefix = ['D_t','D_init_barg'], dtype = float) 
    #X_t=pd.get_dummies(data['t''init_barg'], columns = ['t'], prefix = 'D_t', dtype = float, drop_first=True,  ) 
    X_t = X_t.drop(columns = ['D_t_12','D_init_barg_1']) #drop reference cat


    #SHADOW VALUE: 
    if shadow_value_simple: 
        Shadow_value = data_regress[['delta_log_fam_inc', 'delta_log_wealth', 'log_fam_inc_l', 'log_wealth_l']]
    else:
        Shadow_value = data_regress[['t']]
        #data_test['earbubgs'] = pd.qcut(data['log_wealth'], 10, labels = False) 
        cat = ['log_earnings_w', 'log_earnings_m','log_earnings_w_l', 'log_earnings_m_l', 'inc_share_w', 'inc_share_w_l','log_wealth', 'wealth_F']
        


        for i in cat:
            Shadow_value[i] = pd.qcut(data_regress[i], 10, labels = False, duplicates='raise') 

        Shadow_value['earnings_w_m'] = Shadow_value['log_earnings_w'].astype('str') + '_' + Shadow_value['log_earnings_m'].astype('str')
        Shadow_value['earnings_w_m_l'] = Shadow_value['log_earnings_w_l'].astype('str') + '_' + Shadow_value['log_earnings_m_l'].astype('str')
        Shadow_value['inc_share_n_l'] = Shadow_value['inc_share_w_l'].astype('str') + '_' + Shadow_value['inc_share_w'].astype('str')
        Shadow_value['wealth_n_l'] = Shadow_value['log_wealth'].astype('str') + '_' + Shadow_value['wealth_F'].astype('str')

        Shadow_value = pd.get_dummies(Shadow_value, columns=['earnings_w_m', 'earnings_w_m_l', 'inc_share_n_l','wealth_n_l' ], drop_first = True, dtype = float)


        #Drop if less than two
        Shadow_value = Shadow_value.loc[:,(Shadow_value.sum()>2 )]
        Shadow_value = Shadow_value.drop(columns = ['t','log_earnings_w', 'log_earnings_m','log_earnings_w_l', 'log_earnings_m_l', 'inc_share_w', 'inc_share_w_l','log_wealth','wealth_F'])



    df = data_regress.drop(columns = ['t','init_barg','log_earnings_w', 'log_earnings_m','log_earnings_w_l', 'log_earnings_m_l', 'inc_share_w', 'inc_share_w_l','log_wealth', 'wealth_F',  f'y_{gender}', 'idx','delta_log_fam_inc', 'log_fam_inc_l', 'log_wealth_l'])

    #REGRESS
    y  = data_regress[f'y_{gender}']
    x = pd.concat([df,  X_t , Shadow_value], axis=1 )
    x = x.T.drop_duplicates().T #drop duplicates
    x = sm.add_constant(x)  
    #result = sm.OLS(y,x).fit() 
    result = sm.OLS(y,x).fit().get_robustcov_results(cov_type = 'cluster', groups = data_regress['idx'])
    N = result.nobs
    #SAVE WALD TEST
    Wald_FC = result.wald_test('(wage_shock_l=0, wage_shock_l2=0,wage_shock_j=0,wage_shock_j_l=0, wage_shock_j_l2=0, BMI=0, BMI_l=0, BMI_l2=0,BMI_j=0,BMI_j_l=0, BMI_j_l2=0,D_init_barg_0=0,D_init_barg_2=0)', use_f = True)
    Wald_NC = result.wald_test('(wage_shock_l=0, wage_shock_l2=0,               wage_shock_j_l=0, wage_shock_j_l2=0,        BMI_l=0, BMI_l2=0,        BMI_j_l=0, BMI_j_l2=0,D_init_barg_0=0,D_init_barg_2=0)', use_f = True)

    if print_reg:
        print(result.summary())


        #test for full commitment
        print(f' Test for full commitment')
        print(Wald_FC )

        #test for no commitment
        print(f' Test for no commitment')
        print(Wald_NC)

    #FOR MAN!!

    return  data_regress, Wald_FC, Wald_NC, N

