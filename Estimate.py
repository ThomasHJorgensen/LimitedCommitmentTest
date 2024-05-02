
#import
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import statsmodels.api as sm

def create_data(model,start_p = 1, end_p = 4, to_xl = False, name_xl = 'simulated_data',path='output/', yerror = "norm", scale_st = 2.0):
    
    #unpack
    par = model.par 
    sim = model.sim 

    #update stockastic elements.
    shape_sim = (par.simN,par.T)
    sim.draw_love = par.sigma_love * np.random.normal(size=shape_sim)
    sim.draw_Kw = par.sigma_K * np.random.normal(size=shape_sim)
    sim.draw_Km = par.sigma_K * np.random.normal(size=shape_sim)

    #simulate the data
    model.simulate()

    #HOURS 
    hours_w_orig = model.sim.labor_w 
    hours_m_orig = model.sim.labor_m 
    hours_w_std = np.nanstd(hours_w_orig)
    hours_m_std = np.nanstd(hours_m_orig)

    #measurement error in y (hours)
    if yerror == "norm":
        #norm distributed
        hours_w     =  hours_w_orig  + np.random.normal(scale=scale_st*hours_w_std, size=shape_sim) 
        hours_m     =  hours_m_orig  + np.random.normal(scale=scale_st*hours_m_std, size=shape_sim) 
    elif yerror == "uni":
        #uniform distributed
        hours_w     =  hours_w_orig  + np.random.uniform(-scale_st*hours_w_std,    scale_st*hours_w_std, size=shape_sim) 
        hours_m     =  hours_m_orig  + np.random.uniform(-scale_st*hours_m_std,    scale_st*hours_m_std, size=shape_sim)
    elif yerror == "none":
        #no measurement error
        hours_w     =  hours_w_orig   
        hours_m     =  hours_m_orig  
    else:
        raise Exception("Yerror must be norm, uni or none")

    #ensure  hours is between 0 and 1
    hours_w     = np.maximum(0.0,np.minimum(1.0,hours_w))
    hours_m     = np.maximum(0.0,np.minimum(1.0,hours_m))

    #Find the data
    wage_w      =  np.exp(model.par.wage_const_w +model.par.wage_K_w*model.sim.Kw) 
    wage_m      =  np.exp(model.par.wage_const_m+model.par.wage_K_m* model.sim.Km)  
    y_w         =  wage_w*hours_w_orig
    y_m         =  wage_m*hours_m_orig
    init_barg   =  model.sim.init_distr
    Z_w         =  model.sim.Zw 
    Z_m         =  model.sim.Zm 
    wealth      =  model.sim.A


    #Save the data in a data frame
    data_nu = {}
    data   = pd.DataFrame()
   
    #insert in dataframe
    for i in range(start_p, end_p): #use some periods in the middle of the simluation
        data_nu[i] = pd.DataFrame({
            'idx': range(1,model.par.simN+1) ,
            't' : i,
            'wealth' : wealth[:,i] ,
            'couple': model.sim.couple[:,i],
            'hours_w': hours_w[:,i],
            'hours_m': hours_m[:,i],
            'cons': model.sim.cons_w[:,i],
            'wage_w': wage_w[:,i],
            'wage_m': wage_m[:,i],
            'earnings_w': y_w[:,i],
            'earnings_m': y_m[:,i],
            'BMI_w': Z_w[:,i],
            'BMI_m': Z_m[:,i],
            'omega_w':  model.sim.draw_Kw_temp[:,i],
            'omega_m':  model.sim.draw_Km_temp[:,i],
            'init_barg': init_barg,
            'value'    : model.sim.value[:,i],
            'util' : model.sim.util[:,i],
            'barganing': model.sim.power[:,i],
            'Love': model.sim.love[:,i],
            'exp_w': model.sim.exp_w[:,i],
            'exp_m': model.sim.exp_m[:,i], 
            'wage_w_orig': wage_w[:,i],
            'wage_m_orig': wage_m[:,i],
            'hours_w_orig':   hours_w_orig[:,i],  
            'hours_m_orig':   hours_m_orig[:,i],  
            'wealth_orig' :    wealth[:,i]

        })

        
        #collect the data
        data = pd.concat([data,data_nu[i]] )


    #sort data
    data = data.sort_values(by =['idx','t'])
    if to_xl: 
        data.to_excel(f'{path}{name_xl}.xlsx')


    return data

def create_variable(data, par, print_aux_reg = False):
    data = data.sort_values(by =['idx','t'])

    #Variable I do not use
    data = data.drop(columns = ['util','value','exp_w','exp_m'])


    #change BMI to a 1-2 instead of 0 -1 (so I can take the logarithm)
    data['BMI_w'] = 1 + data['BMI_w'] 
    data['BMI_m'] = 1 + data['BMI_m'] 

    #remove negative values
    data['omega_w'] = 1 + data['omega_w'] 
    data['omega_m'] = 1 + data['omega_m'] 

    
    #remove negative values
    data['Love'] = 1 + data['Love'] 

    #drop if single
    data = data[data['couple']>0.5]
    data = data.drop(columns = ['couple'])

    #drop begining and end of period
    #data = data[data['t']>8]
    #data = data[data['t']<15]
    data = data[data['t']>4]
    data = data[data['t']<16]

    #drop if not paricipating
    data = data[data['hours_w']>0.001]
    data = data[data['hours_m']>0.001]

    #gen faminc and income share
    data['fam_inc'] = data['earnings_w']+data['earnings_m']
    data['inc_share_w'] = data['earnings_w']/data['fam_inc']
    data['inc_share_m'] = data['earnings_m']/data['fam_inc']
    

    #transform with  lag 
    list = ['t','idx','BMI_w', 'BMI_m', 'omega_w', 'omega_m', 'hours_w','hours_m', 'cons','wage_w','wage_m','earnings_w','earnings_m','fam_inc','wealth','inc_share_w','inc_share_m', 'Love']

    data_l = data.loc[:,list]
    data_l['t'] = data_l['t']+1

    data_l2 = data.loc[:,list]
    data_l2['t'] = data_l2['t']+2

    data_l3 = data.loc[:,list]
    data_l3['t'] = data_l3['t']+3

    
    data_F = data.loc[:,['t','idx','wealth']]
    data_F['t'] = data_F['t']-1

    #transfor with logs
    list_log = ['BMI_w', 'BMI_m', 'hours_w','hours_m', 'cons','wage_w','wage_m','earnings_w','earnings_m','fam_inc','wealth','omega_w', 'omega_m', 'Love']
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

    
    #Estimate wage shocks from wage equation
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
 

    #Create variables
    data['delta_omega_w'] = data['log_omega_w']-data['log_omega_w_l']
    data['delta_omega_m'] = data['log_omega_m']-data['log_omega_m_l']    
    data['delta_omega_w_l'] = data['log_omega_w_l']-data['log_omega_w_l2']
    data['delta_omega_m_l'] = data['log_omega_m_l']-data['log_omega_m_l2']    
    data['delta_omega_w_l2'] = data['log_omega_w_l2']-data['log_omega_w_l3']
    data['delta_omega_m_l2'] = data['log_omega_m_l2']-data['log_omega_m_l3']
    data['y_w'] = data['delta_log_hours_w']*data['hours_w_l']
    data['y_m'] = data['delta_log_hours_m']*data['hours_m_l']
    data['control_part_inc_w'] = data['inc_share_m_l']*data['delta_log_earnings_m']
    data['control_part_inc_m'] = data['inc_share_w_l']*data['delta_log_earnings_w']
    data['control_cons'] = data['cons_l']*data['delta_log_cons']

    #keep variable I need
    #data = data[['t','idx','init_barg', 'y_w', 'y_m', 'inc_share_w', 'inc_share_w_l', 'delta_log_wage_w','delta_log_wage_m','delta_log_wage_w_l','delta_log_wage_m_l','delta_log_wage_w_l2','delta_log_wage_m_l2', 'omega_res_w', 'omega_res_m','omega_res_w_l', 'omega_res_m_l','omega_res_w_l2', 'omega_res_m_l2', 'delta_omega_m','delta_omega_w','delta_omega_w_l', 'delta_omega_m_l','delta_omega_w_l2', 'delta_omega_m_l2', 'control_part_inc_w', 'control_part_inc_m', 'control_cons', 'delta_log_wealth','delta_log_wealth_l', 'delta_log_wealth_l2' ,'delta_log_fam_inc', 'log_fam_inc', 'log_wealth', 'log_fam_inc_l', 'log_fam_inc_l2', 'log_wealth_l','log_wealth_l2', 'wealth_F', 'log_earnings_w', 'log_earnings_m', 'log_earnings_w_l', 'log_earnings_m_l', 'delta_log_BMI_w', 'delta_log_BMI_m', 'delta_log_BMI_w_l', 'delta_log_BMI_m_l', 'delta_log_BMI_w_l2', 'delta_log_BMI_m_l2', 'delta_log_Love', 'delta_log_Love_l', 'delta_log_Love_l2']]
    
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


def main_est(data, gender = "w", do_estimate_wage = "est_omega", print_reg = False, shadow_value_simple = 1, do_control_love = False):
    #TODO tillad for at man kun tester på baggrund af en time varying distributinal faktor

    #Find the gender and the gender of the spouse
    if gender == "w":
        spouse = "m"
    elif gender =='m':
        spouse = "w"
    else: 
        raise Exception("Gender must be m or w")

    #Find time varying distirbutional facto
    if do_estimate_wage == "est_omega":
        data['wage_shock']=data[f'omega_res_{gender}']
        data['wage_shock_l']=data[f'omega_res_{gender}_l']
        data['wage_shock_l2']=data[f'omega_res_{gender}_l2']
        data['wage_shock_j']=data[f'omega_res_{spouse}']
        data['wage_shock_j_l']=data[f'omega_res_{spouse}_l']
        data['wage_shock_j_l2']=data[f'omega_res_{spouse}_l2']
    elif do_estimate_wage == "true_omega":
        data['wage_shock']=data[f'delta_omega_{gender}']
        data['wage_shock_l']=data[f'delta_omega_{gender}_l']
        data['wage_shock_l2']=data[f'delta_omega_{gender}_l2']
        data['wage_shock_j']=data[f'delta_omega_{spouse}']
        data['wage_shock_j_l']=data[f'delta_omega_{spouse}_l']
        data['wage_shock_j_l2']=data[f'delta_omega_{spouse}_l2']
    elif do_estimate_wage == "wage":
        data['wage_shock']=data[f'delta_log_wage_{gender}']
        data['wage_shock_l']=data[f'delta_log_wage_{gender}_l']
        data['wage_shock_l2']=data[f'delta_log_wage_{gender}_l2']
        data['wage_shock_j']=data[f'delta_log_wage_{spouse}']
        data['wage_shock_j_l']=data[f'delta_log_wage_{spouse}_l']
        data['wage_shock_j_l2']=data[f'delta_log_wage_{spouse}_l2']
    else: 
        raise Exception("do_estimate_wage must be est_omega, true_omega, wage")

  
    data['BMI'] = data[f'delta_log_BMI_{gender}']
    data['BMI_l'] = data[f'delta_log_BMI_{gender}_l']
    data['BMI_l2'] = data[f'delta_log_BMI_{gender}_l2']
    data['BMI_j'] = data[f'delta_log_BMI_{spouse}']
    data['BMI_j_l'] = data[f'delta_log_BMI_{spouse}_l']
    data['BMI_j_l2'] = data[f'delta_log_BMI_{spouse}_l2']

    
    
    
    data_regress = data[['t','init_barg','log_earnings_w', 'log_earnings_m','log_earnings_w_l', 'log_earnings_m_l', 'inc_share_w', 'inc_share_w_l','log_wealth', 'wealth_F',  f'y_{gender}', 'idx', 'wage_shock','wage_shock_l','wage_shock_l2','wage_shock_j','wage_shock_j_l','wage_shock_j_l2','BMI','BMI_l','BMI_l2','BMI_j','BMI_j_l','BMI_j_l2',f'control_part_inc_{gender}','control_cons','delta_log_wealth','delta_log_wealth_l','delta_log_wealth_l2','delta_log_Love','delta_log_Love_l','delta_log_Love_l2','delta_log_fam_inc', 'log_fam_inc' , 'log_fam_inc_l', 'log_fam_inc_l2', 'log_wealth_l', 'log_wealth_l2']]

    #DROP NAN
    data_regress = data_regress.dropna() # det ser ud som om den ikke fjerner nogen

    #PREPARE T
    X_t=pd.get_dummies(data_regress[['t', 'init_barg']], columns = ['t','init_barg'], prefix = ['D_t','D_init_barg'], dtype = float) 
    #X_t=pd.get_dummies(data['t''init_barg'], columns = ['t'], prefix = 'D_t', dtype = float, drop_first=True,  ) 
    X_t = X_t.drop(columns = ['D_t_12','D_init_barg_1']) #drop reference cat


    #SHADOW VALUE: 
    if shadow_value_simple == 1: 
        Shadow_value = data_regress[['delta_log_fam_inc', 'delta_log_wealth', 'log_fam_inc_l', 'log_wealth_l']]

    
    elif shadow_value_simple == 2:

        Shadow_value = data_regress[['t']]
        cat = ['delta_log_fam_inc', 'delta_log_wealth', 'log_fam_inc_l', 'log_wealth_l']
    
        for i in cat:
            Shadow_value[i] = pd.qcut(data_regress[i], 100, labels = False, duplicates='raise') 

        Shadow_value = pd.get_dummies(Shadow_value, columns=['delta_log_fam_inc', 'delta_log_wealth', 'log_fam_inc_l', 'log_wealth_l' ], drop_first = True, dtype = float)


        #Drop if less than two
        Shadow_value = Shadow_value.loc[:,(Shadow_value.sum()>2 )]
        Shadow_value = Shadow_value.drop(columns = ['t'])




    elif shadow_value_simple == 3:
        
        Shadow_value = data_regress[['t']]
        cat = ['delta_log_fam_inc', 'delta_log_wealth', 'log_fam_inc_l', 'log_wealth_l']
    
        for i in cat:
            Shadow_value[i] = pd.qcut(data_regress[i], 200, labels = False, duplicates='raise') 

        Shadow_value = pd.get_dummies(Shadow_value, columns=['delta_log_fam_inc', 'delta_log_wealth', 'log_fam_inc_l', 'log_wealth_l' ], drop_first = True, dtype = float)


        #Drop if less than two
        Shadow_value = Shadow_value.loc[:,(Shadow_value.sum()>2 )]
        Shadow_value = Shadow_value.drop(columns = ['t'])



    elif shadow_value_simple == 4:
        
        Shadow_value = data_regress[['t']]
        cat = ['delta_log_fam_inc', 'delta_log_wealth', 'log_fam_inc_l', 'log_wealth_l']
    
        for i in cat:
            Shadow_value[i] = pd.qcut(data_regress[i], 50, labels = False, duplicates='raise') 

        
        Shadow_value['delta_log_fam_inc_t'] = Shadow_value['delta_log_fam_inc'].astype('str') + '_' + Shadow_value[ 't'].astype('str')
        Shadow_value['delta_log_wealth_t'] = Shadow_value['delta_log_wealth'].astype('str') + '_' + Shadow_value[ 't'].astype('str')
        Shadow_value['log_fam_inc_l_t'] = Shadow_value['log_fam_inc_l'].astype('str') + '_' + Shadow_value[ 't'].astype('str')
        Shadow_value['log_wealth_l_t'] = Shadow_value['log_wealth_l'].astype('str') + '_' + Shadow_value[ 't'].astype('str')

        Shadow_value = pd.get_dummies(Shadow_value, columns=['delta_log_fam_inc_t', 'delta_log_wealth_t', 'log_fam_inc_l_t', 'log_wealth_l_t' ], drop_first = True, dtype = float)



        #Drop if less than two
        Shadow_value = Shadow_value.loc[:,(Shadow_value.sum()>2 )]
        Shadow_value = Shadow_value.drop(columns = ['t', 'delta_log_fam_inc', 'delta_log_wealth', 'log_fam_inc_l', 'log_wealth_l'])

    else:
        raise Exception("shadow_value_simple must be 1, 2, 3 or 4")


    df = data_regress.drop(columns = ['t','init_barg','log_earnings_w', 'log_fam_inc' , 'log_earnings_m','log_earnings_w_l', 'log_earnings_m_l', 'inc_share_w', 'inc_share_w_l','log_wealth', 'wealth_F',  f'y_{gender}', 'idx','delta_log_fam_inc', 'log_fam_inc_l', 'log_wealth_l', 'log_fam_inc_l2', 'log_wealth_l2', 'delta_log_wealth'])
    if not do_control_love:
         df = df.drop(columns = ['delta_log_Love','delta_log_Love_l','delta_log_Love_l2'])
    

    #REGRESS
    y  = data_regress[f'y_{gender}']
    x = pd.concat([df,  X_t , Shadow_value], axis=1 )
    #x = x.T.drop_duplicates().T #drop duplicates
    x = sm.add_constant(x)  
    #result = sm.OLS(y,x).fit() 
    result = sm.OLS(y,x).fit().get_robustcov_results(cov_type = 'cluster', groups = data_regress['idx'])
    N = result.nobs
    #SAVE WALD TEST
    Wald_FC = result.wald_test('(wage_shock_l=0, wage_shock_l2=0, wage_shock_j=0, wage_shock_j_l=0, wage_shock_j_l2=0, BMI=0, BMI_l=0, BMI_l2=0,BMI_j=0, BMI_j_l=0, BMI_j_l2=0, D_init_barg_0=0, D_init_barg_2=0)', use_f = True)
    Wald_NC = result.wald_test('(wage_shock_l=0, wage_shock_l2=0,                wage_shock_j_l=0, wage_shock_j_l2=0,        BMI_l=0, BMI_l2=0,         BMI_j_l=0, BMI_j_l2=0, D_init_barg_0=0, D_init_barg_2=0)', use_f = True)
    Wald_FC_noW = result.wald_test('(                                                                                BMI=0, BMI_l=0, BMI_l2=0,BMI_j=0, BMI_j_l=0, BMI_j_l2=0, D_init_barg_0=0, D_init_barg_2=0)', use_f = True)
    Wald_NC_noW = result.wald_test('(                                                                                       BMI_l=0, BMI_l2=0,         BMI_j_l=0, BMI_j_l2=0, D_init_barg_0=0, D_init_barg_2=0)', use_f = True)


    if print_reg:
        print(result.summary())


        #test for full commitment
        print(f' Test for full commitment')
        print(Wald_FC )

        #test for no commitment
        print(f' Test for no commitment')
        print(Wald_NC)

    #FOR MAN!!

    return  data_regress, Wald_FC, Wald_NC, Wald_FC_noW, Wald_NC_noW, N

