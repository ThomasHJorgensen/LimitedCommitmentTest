
#import
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import statsmodels.api as sm

def create_data_new(model,start_p = 1, end_p = 4, to_xl = False, name_xl = 'simulated_data'):
    
    par = model.par 
    sim = model.sim 

    #update stockastic elements.
    shape_sim = (par.simN,par.T)
    sim.draw_love = par.sigma_love * np.random.normal(size=shape_sim)
    sim.draw_Kw = np.exp(-0.5*par.sigma_K**2 + par.sigma_K*np.random.normal(size=shape_sim))
    sim.draw_Km = np.exp(-0.5*par.sigma_K**2 + par.sigma_K*np.random.normal(size=shape_sim))

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
    init_barg   =  model.sim.init_Kw > model.sim.init_Km
    Z_w         = 1 #TODO: Include Z values
    Z_m         = 1 #TODO: Include Z values


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
            'BMI_w': Z_w,
            'BMI_m': Z_m,
            'omega_w':  model.sim.draw_Kw[:,i],
            'omega_m':  model.sim.draw_Km[:,i],
            'init_barg': init_barg,
            'value'    : model.sim.value[:,i],
            'util' : model.sim.util[:,i],
            'barganing': model.sim.power[:,i]
        })

        

        #collect the data
        data = pd.concat([data,data_nu[i]] )


    #drop if single
    data= data.drop(data[data.couple==0].index) 


    #sort data
    data = data.sort_values(by =['idx','t'])
    if to_xl: 
        data.to_excel(f'{name_xl}.xlsx')

    #create variable
    #data = create_variable(data)


    return data

def create_data(model,start_p = 1, end_p = 4, to_xl = False, name_xl = 'simulated_data'):
    
    par = model.par 
    sim = model.sim 

    #update stockastic elements.
    shape_sim = (par.simN,par.T)
    sim.draw_love = par.sigma_love * np.random.normal(size=shape_sim)
    sim.draw_Kw = np.exp(-0.5*par.sigma_K**2 + par.sigma_K*np.random.normal(size=shape_sim))
    sim.draw_Km = np.exp(-0.5*par.sigma_K**2 + par.sigma_K*np.random.normal(size=shape_sim))

    #simulate the data
    model.simulate()

    #Save the data in a data frame
    data_w = {}
    data_m = {}
    data   = pd.DataFrame()

    #TODO: save it while simulating
    wage_w      =  np.exp(model.par.wage_const_w +model.par.wage_K_w* model.sim.Kw)  
    y_w         =  wage_w*model.sim.labor_w
    wage_m      =  np.exp(model.par.wage_const_m+model.par.wage_K_m* model.sim.Km)
    y_m         =  wage_m*model.sim.labor_m
    init_barg   =  model.sim.init_Kw > model.sim.init_Km
    Z_w         = 1 #TODO: Include Z values
    Z_m         = 1 #TODO: Include Z values

    for i in range(start_p, end_p): #use some periods in the middle of the simluation

        # WOMAN
        #value this period and last period, 
        data_w[i] = pd.DataFrame({
            'idx': range(1,model.par.simN+1) ,
            't' : i,
            'woman' : 1,
            'A' : model.sim.A[:,i] ,
            'couple': model.sim.couple[:,i],
            'Labor': model.sim.labor_w[:,i],
            'Labor_j': model.sim.labor_m[:,i],
            'cons': model.sim.cons_w[:,i],
            'wage': wage_w[:,i],
            'wage_j': wage_m[:,i],
            'y': y_w[:,i],
            'y_j': y_m[:,i],
            'Z': Z_w,
            'Z_j': Z_m,
            'init_barg': init_barg,
            'A_1' : model.sim.A[:,i-1] ,  
            'Labor_1': model.sim.labor_w[:,i-1],
            'Labor_j_1': model.sim.labor_m[:,i-1],
            'cons_1': model.sim.cons_w[:,i-1],
            'wage_1': wage_w[:,i-1],
            'wage_j_1': wage_m[:,i-1],
            'y_1': y_w[:,i-1],
            'y_j_1': y_m[:,i-1],
            'Z_1': Z_w,
            'Z_j_1': Z_m
        })

        
        # MAN
        data_m[i] = pd.DataFrame({
            'idx': range(model.par.simN+1,2*model.par.simN+1) ,
            't' : i,
            'woman' : 0,
            'A' : model.sim.A[:,i] ,
            'couple': model.sim.couple[:,i],
            'Labor': model.sim.labor_m[:,i],
            'Labor_j': model.sim.labor_w[:,i],
            'cons': model.sim.cons_m[:,i],
            'wage': wage_m[:,i],
            'wage_j': wage_w[:,i],
            'y': y_m[:,i],
            'y_j': y_w[:,i],
            'Z': Z_m,
            'Z_j': Z_w,
            'init_barg': init_barg,
            'A_1' : model.sim.A[:,i-1] ,
            'Labor_1': model.sim.labor_m[:,i-1],
            'Labor_j_1': model.sim.labor_w[:,i-1],
            'cons_1': model.sim.cons_m[:,i-1],
            'wage_1': wage_m[:,i-1],
            'wage_j_1': wage_w[:,i-1],
            'y_1': y_m[:,i-1],
            'y_j_1': y_w[:,i-1],
            'Z_1': Z_m,
            'Z_j_1': Z_w      
        
        }) 

        #collect the data
        data = pd.concat([data,data_w[i], data_m[i] ] )


    #drop if single
    data= data.drop(data[data.couple==0].index) #few obs dropped?


    #sort data
    data = data.sort_values(by =['idx','t'])
    if to_xl: 
        data.to_excel(f'{name_xl}.xlsx')

    #create variable
    data = create_variable(data)


    return data


def create_variable(data):
    list = ['Labor', 'Labor_j', 'wage' , 'wage_j', 'y', 'y_j', 'Z', 'Z_j', 'A', 'cons']

    for val in list:
        val2  =  val + '_1'
        val_name = 'log_' + val
        val_name2 = 'log_' + val + '_1'
        name       = 'delta_log_' + val
        data = data.replace({val: 0}, np.nan)
        data[val_name] = np.log(data[val])
        data[val_name2] = np.log(data[val2])    
        data[name] = data[val_name].sub(data[val_name2]) 

    
    data['Labor_inv_1'] = data.apply(lambda row: 1/row['Labor_1'], axis=1)
    #data['Labor_inv'] = data.apply(lambda row: 1/row['Labor'], axis=1)
    data['total_inc']  = data['y'].add(data['y_j']) 
    data['inc_share_j'] =  data['y_j'].div(data['total_inc'])
    #data['Laborinv_t'] =  data['Labor_inv'].mul(data['t'])
    data['Laborinv_1_t'] =  data['Labor_inv_1'].mul(data['t'])
    data['Laborinv_1_woman'] =  data['Labor_inv_1'].mul(data['woman'])

    return data

def aux_est(data, print_reg = False):

    # drop nan
    data_regress = data[['idx','t', 'Labor_inv_1','Laborinv_1_t', 'Laborinv_1_woman', 'delta_log_Labor', 'delta_log_wage_j','delta_log_wage','woman']].dropna()



    #Step1 find the residuals from hours equation 

    #defining the variables 
    x = data_regress[['Labor_inv_1', 'Laborinv_1_t', 'Laborinv_1_woman']]    #the last two is from wage observable
    y  = data_regress['delta_log_Labor']
    #x = sm.add_constant(x) #Noconstant

    #performing the regression and fitting the model
    result = sm.OLS(y,x).fit()
    if print_reg:
        print('Residuals from hours equation')
        print(result.summary())

    #find the residuals
    data_regress['uhat'] = result.resid



    #Step2 find the residuals from the wage regression, (Note: we can find it directly from our solution)
    x = data_regress[['t', 'woman']]
    y  = data_regress['delta_log_wage']
    x = sm.add_constant(x) 
    result = sm.OLS(y,x).fit()
    if print_reg:
        print('Residuals from own wage equation')
        print(result.summary())
    data_regress['omega'] = result.resid


    x = data_regress[['t', 'woman']]
    y  = data_regress['delta_log_wage_j']
    x = sm.add_constant(x) 
    result = sm.OLS(y,x).fit()
    if print_reg:
        print('Residuals from partners wage equation')
        print(result.summary())
    data_regress['omega_j'] = result.resid



    # merge the residuals to the main data
    data_regress = data_regress[['idx', 't','uhat', 'omega', 'omega_j']]
    data = data.merge(data_regress, on = ['idx', 't'])

    return data




def main_est(data, print_reg = False):


    #prepare for main regression
    #Find the lagged value. Note: it works because single is absorbing, when people leave the data they will not come back
    list = ['omega','omega_j','delta_log_Z', 'delta_log_Z_j', 'delta_log_y','delta_log_y_j',  'delta_log_A','inc_share_j']
    data = data.sort_values(by =['idx','t']) 
    data['idx_1'] = data['idx'].shift(periods =1 )

    for val in list:
        val_name = val + '_1'
        data[val_name] = data[val].shift(periods =1 ) 
        data.loc[data['idx_1'] !=  data['idx'],val_name] = np.nan


    data['X3'] =  data['omega'].mul(data['Labor_inv_1'])
    data['X4'] =  data['omega_j'].mul(data['Labor_inv_1'])
    data['X5'] =  data['omega_1'].mul(data['Labor_inv_1'])
    data['X6'] =  data['omega_j_1'].mul(data['Labor_inv_1'])

    data['X7a'] =  data['delta_log_Z'].mul(data['Labor_inv_1'])
    data['X7b'] =  data['delta_log_Z_1'].mul(data['Labor_inv_1'])
    data['X8a'] =  data['delta_log_Z_j'].mul(data['Labor_inv_1'])
    data['X8b'] =  data['delta_log_Z_j_1'].mul(data['Labor_inv_1'])

    data['X9'] = data['init_barg'].mul(data['Labor_inv_1'])

    data['X10'] = data['delta_log_y'].mul(data['Labor_inv_1'])
    data['X11'] = data['delta_log_A'].mul(data['Labor_inv_1'])
    data['X12'] = data['delta_log_y_1'].mul(data['Labor_inv_1'])
    data['X13'] = data['delta_log_A_1'].mul(data['Labor_inv_1'])

    data['X14'] = data['inc_share_j_1'].mul(data['delta_log_y_j_1']).mul(data['Labor_inv_1'])
    data['X15'] = data['cons_1'].mul(data['delta_log_cons']).mul(data['Labor_inv_1'])
    #x16: the same as x13, when we only have two periods


    #data_regress = data[['X3','X4','X5','X6','X7a','X7b','X8a','X8b','X9','X10','X11','X12','X13','X14','X15','uhat']] #with Z
    data_regress = data[['X3','X4','X5','X6','X9','X10','X11','X12','X13','X14','X15','uhat']] #without Z


    #drop nan 
    data_regress = data_regress.dropna()
    #print(data_regress)


    #Run main regression
    #x = data[['X3','X4','X5','X6','X7a','X7b','X8a','X8b','X9','X10','X11','X12','X13','X14','X15']] #with Z
    x = data_regress[['X3','X4','X5','X6','X9','X10','X11','X12','X13','X14','X15']] #without Z
    y  = data_regress['uhat']
    #noconst

    result = sm.OLS(y,x).fit() #TODO: use correct standard errors

    Wald_FC = result.wald_test('(X4=0, X5=0, X6=0, X9 =0)', use_f = False) #TODO FutureWarning: The behavior of wald_test will change after 0.14 to returning scalar test statistic values. To get the future behavior now, set scalar to True. 
    Wald_NC = result.wald_test('(X5=0, X6=0, X9 =0)', use_f = False)

    if print_reg:
        print(result.summary())


        #test for full commitment* TODO CHECK IF IT IS THE CORRECT variable that is tested! 
        print(f' Test for full commitment')
        #print(result.wald_test('(X4=0, X5=0, X6=0, X7a=0, X7b=0, X8a=0, X8b=0 , X9 =0)', use_f = False)) #with Z
        print(Wald_FC )

        #test for no commitment* these should be zero
        print(f' Test for no commitment')
        #print(result.wald_test('(X5=0, X6=0, X7a=0, X7b=0, X8a=0, X8b=0 , X9 =0)', use_f = False)) #with Z
        print(Wald_NC)


    return  data_regress, Wald_FC, Wald_NC

