
// functions for solving model for singles.
#ifndef MAIN
#define SIMULATE
#include "myheader.h"
#endif

namespace sim {

    double limited_commitment(int t,int idx_sol, double power_lag,double Vw_couple,double Vm_couple,double Vw_single,double Vm_single, double love,double A_lag,double Aw_lag,double Am_lag,double Kw, double Km, sol_struct* sol,par_struct* par){
        // check participation constraints
        double power = -1.0; // initialize as divorce

        if ((Vw_couple>=Vw_single) & (Vm_couple>=Vm_single)){
            power = power_lag;

        } else {

            // determine which partner is unsatisfied
            double* V_power_vec = new double[par->num_power];
            double* V_remain_couple;
            double* V_remain_couple_partner;
            double V_single;
            double V_single_partner;

            bool flip = false;
            double* grid_power;
            if ((Vm_couple>=Vm_single)){ // woman wants to leave

                V_remain_couple = sol->Vw_remain_couple;
                V_remain_couple_partner = sol->Vm_remain_couple;

                V_single = Vw_single;
                V_single_partner = Vm_single;

                flip = false;
                grid_power = par->grid_power;

            } else { // man wants to leave

                V_remain_couple = sol->Vm_remain_couple;
                V_remain_couple_partner = sol->Vw_remain_couple;

                V_single = Vm_single;
                V_single_partner = Vw_single;

                flip = true;
                grid_power = par->grid_power_flip;

            }

            // find relevant values of remaining a couple at all other states than power 
            int j_love = tools::binary_search(0,par->num_love,par->grid_love,love); 
            int j_A = tools::binary_search(0,par->num_A,par->grid_A,A_lag);
            int j_Kw = tools::binary_search(0,par->num_K,par->grid_K,Kw);
            int j_Km = tools::binary_search(0,par->num_K,par->grid_K,Km);
            for (int iP=0; iP<par->num_power; iP++){ 
                int idx; 
                if (flip){
                    idx = index::couple(t,par->num_power-1 - iP,0,0,0,0,par); 
                } else {
                    idx = index::couple(t,iP,0,0,0,0,par); 
                }
                V_power_vec[iP] = tools::_interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K, par->num_love,par->num_A,par->num_K,par->num_K, &V_remain_couple[idx],love,A_lag,Kw,Km,j_love,j_A,j_Kw,j_Km);
            }
            

            // interpolate the power based on the value of single to find indifference-point. (flip the axis)
            power = tools::interp_1d(V_power_vec, par->num_power, grid_power, V_single);
            delete V_power_vec;

            if((power<0.0)|(power>1.0)){ // divorce
                return -1.0;
            }

            // calculate value for the other partner at this level of power
            int j_power = tools::binary_search(0,par->num_power,par->grid_power,power);
            double V_partner = tools::_interp_5d(par->grid_power,par->grid_love,par->grid_A,par->grid_K,par->grid_K, par->num_power,par->num_love,par->num_A,par->num_K,par->num_K, &V_remain_couple_partner[idx_sol],power,love,A_lag,Kw,Km,j_power,j_love,j_A,j_Kw,j_Km);
            double S_partner = couple::calc_marital_surplus(V_partner,V_single_partner,par);
            
            // check if partner is happy. if not divorce
            if(S_partner<0.0){
                power = -1.0;  
            }

        }

        return power;

    } // limited_commitment

    
    double nash(int t,int idx_sol, double Vw_single,double Vm_single, double love,double A_lag,double Aw_lag,double Am_lag,double Kw, double Km, sol_struct* sol,par_struct* par){
        
        double power = -1.0; // initialize as divorce

        // find (discrete) max amd store vectors of surpluses
        double* Sm = new double[par->num_power];
        double* Sw = new double[par->num_power];

        double obj_max = -1.0e10;
        int iP_max = -1;

        int j_love = tools::binary_search(0,par->num_love,par->grid_love,love); 
        int j_A = tools::binary_search(0,par->num_A,par->grid_A,A_lag);
        int j_Kw = tools::binary_search(0,par->num_K,par->grid_K,Kw);
        int j_Km = tools::binary_search(0,par->num_K,par->grid_K,Km);
        for (int iP=0; iP<par->num_power; iP++){
            // value of remaining a couple. 
            int idx_sol = index::couple(t,iP,0,0,0,0,par); 
            double Vw_couple,Vm_couple;
            tools::_interp_4d_2out(&Vw_couple,&Vm_couple,par->grid_love,par->grid_A,par->grid_K,par->grid_K, par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx_sol],&sol->Vm_remain_couple[idx_sol],love,A_lag,Kw,Km,j_love,j_A,j_Kw,j_Km);
            
            // marital surplus
            Sw[iP] = couple::calc_marital_surplus(Vw_couple,Vw_single,par);
            Sm[iP] = couple::calc_marital_surplus(Vm_couple,Vm_single,par);

            if((Sw[iP]>0.0) & (Sm[iP]>0.0)){
            
                double obj_now = sqrt(Sw[iP]) * sqrt(Sm[iP]);
                
                if(obj_now>obj_max){
                    obj_max = obj_now;
                    iP_max = iP;
                }
            }
        }

        // continuous nash bargaining using discrete as starting point (if anything feasable)
        if(iP_max>-1){
            double minf=0.0;
            int const dim = 1;
            double lb[dim],ub[dim],y[dim];
            
            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT

            // bounds
            lb[0] = 0.0;
            ub[0] = 1.0;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            nlopt_set_ftol_rel(opt,1.0e-5);
            nlopt_set_xtol_rel(opt,1.0e-5);

            bargaining::solver_struct_nash* solver_data = new bargaining::solver_struct_nash;
            solver_data->par = par;
            solver_data->Sw = Sw;
            solver_data->Sm = Sm;
            nlopt_set_min_objective(opt, bargaining::obj_nash, solver_data); 

            // optimize
            y[0] = par->grid_power[iP_max];
            nlopt_optimize(opt, y, &minf); 

            // destroy optimizer
            nlopt_destroy(opt);

            power = y[0];

        } else {
            
            power = -1.0; // divorce
        }

        // delete memory
        delete Sw;
        delete Sm;

        // return power
        return power;

    } // nash


    double update_power(int t, double power_lag, double love,double A_lag,double Aw_lag,double Am_lag,double Kw, double Km, sol_struct* sol, par_struct* par,int bargaining){
        
        // value of transitioning into singlehood
        int idx_single = index::single(t,0,0,par);
        double Vw_single = tools::interp_2d(par->grid_Aw,par->grid_K,par->num_A,par->num_K,&sol->Vw_trans_single[idx_single],Aw_lag,Kw);
        double Vm_single = tools::interp_2d(par->grid_Am,par->grid_K,par->num_A,par->num_K,&sol->Vm_trans_single[idx_single],Am_lag,Km);

        // value of remaining a couple with current power.
        int idx_sol = index::couple(t,0,0,0,0,0,par); 
        double Vw_couple,Vm_couple;
        tools::interp_5d_2out(&Vw_couple,&Vm_couple,par->grid_power,par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx_sol],&sol->Vm_remain_couple[idx_sol], power_lag,love,A_lag,Kw,Km);
        
        
        if(bargaining==0){ // no bargaining: full commitment
            if((Vw_couple<Vw_single) | (Vm_couple<Vm_single)){
                return -1.0;
            } else {
                return power_lag;
            }

        } else if(bargaining==2){ // NASH

            return nash(t,idx_sol,Vw_single,Vm_single, love,A_lag,Aw_lag,Am_lag,Kw,Km,sol,par);
            

        } else { // limited commitment

            return limited_commitment(t,idx_sol,power_lag,Vw_couple,Vm_couple,Vw_single,Vm_single, love,A_lag,Aw_lag,Am_lag,Kw,Km,sol,par);
            
        } // bargaining check

    } // update_power


    void model(sim_struct *sim, sol_struct *sol, par_struct *par){
    
        // pre-compute intra-temporal optimal allocation
        #pragma omp parallel num_threads(par->threads)
        {
            #pragma omp for
            for (int i=0; i<par->simN; i++){
                for (int t=0; t < par->T; t++){
                    int it = index::index2(i,t,par->simN,par->T);
                    int it_1 = index::index2(i,t-1,par->simN,par->T);
                    int it1 = index::index2(i,t+1,par->simN,par->T);

                    double A_lag = sim->init_A[i];
                    double Aw_lag = sim->init_Aw[i];
                    double Am_lag = sim->init_Am[i];
                    bool couple_lag = sim->init_couple[i];
                    double power_lag = par->grid_power[sim->init_power_idx[i]];
                    int bargaining = par->bargaining;

                    // state variables
                    if (t==0){
                        
                        // in other periods these states will be updated in the end of the period below
                        sim->love[it] = sim->init_love[i];
                        sim->Kw[it] = sim->init_Kw[i];
                        sim->Km[it] = sim->init_Km[i];

                        bargaining = 2; // NASH in initial period

                    } else {
                        A_lag = sim->A[it_1];
                        Aw_lag = sim->Aw[it_1];
                        Am_lag = sim->Am[it_1];
                        couple_lag = sim->couple[it_1];
                        power_lag = sim->power[it_1];

                        bargaining = par->bargaining; // whatever specified in the remaining periods
                        
                    } 

                    
                    // first check if they want to remain together and what the bargaining power will be if they do.
                    double power;
                    if (couple_lag) {

                        power = update_power(t,power_lag,sim->love[it],A_lag,Aw_lag,Am_lag,sim->Kw[it],sim->Km[it],sol,par,bargaining);

                        if (power < 0.0) { // divorce is coded as -1
                            sim->couple[it] = false;

                        } else {
                            sim->couple[it] = true;
                        }

                    } else { // remain single
                        sim->couple[it] = false;
                    }

                    // update behavior
                    if (sim->couple[it]){
                        int idx_sol = index::couple(t,0,0,0,0,0,par);

                        // optimal labor supply and consumption
                        tools::interp_5d_2out(&sim->labor_w[it],&sim->labor_m[it], par->grid_power,par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K, &sol->labor_w_remain_couple[idx_sol],&sol->labor_m_remain_couple[idx_sol], power,sim->love[it],A_lag,sim->Kw[it],sim->Km[it]);
                        
                        double resources = couple::resources(sim->labor_w[it],sim->labor_m[it],A_lag,sim->Kw[it],sim->Kw[it],par); 
                        double cons = tools::interp_5d(par->grid_power,par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K, &sol->cons_w_remain_couple[idx_sol], power,sim->love[it],A_lag,sim->Kw[it],sim->Kw[it]); // same for men and women in remain couple
                        cons = MIN(cons,resources); // cannot borrow. This removes small numerical violations
                        sim->cons_w[it] = cons;
                        sim->cons_m[it] = cons;

                        // update end-of-period states
                        sim->A[it] = resources - cons;
                        if(t<par->T-1){
                            sim->love[it1] = sim->love[it] + sim->draw_love[it1];
                            sim->Kw[it1] = utils::K_bar(sim->Kw[it],sim->labor_w[it],par) * sim->draw_Kw[it1];
                            sim->Km[it1] = utils::K_bar(sim->Km[it],sim->labor_m[it],par) * sim->draw_Km[it1];
                        }

                        // in case of divorce
                        sim->Aw[it] = par->div_A_share * sim->A[it];
                        sim->Am[it] = (1.0-par->div_A_share) * sim->A[it];

                        // sim->power_idx[it] = power_idx;
                        sim->power[it] = power;

                    } else { // single
                        
                        // pick relevant solution for single, depending on whether just became single
                        int idx_sol_single = index::single(t,0,0,par); 
                        double *labor_single_w = &sol->labor_w_trans_single[idx_sol_single];
                        double *cons_single_w = &sol->cons_w_trans_single[idx_sol_single];
                        double *labor_single_m = &sol->labor_m_trans_single[idx_sol_single];
                        double *cons_single_m = &sol->cons_m_trans_single[idx_sol_single];
                        if (power_lag<0){
                            labor_single_w = &sol->labor_w_single[idx_sol_single];
                            cons_single_w = &sol->cons_w_single[idx_sol_single];
                            labor_single_m = &sol->labor_m_single[idx_sol_single];
                            cons_single_m = &sol->cons_m_single[idx_sol_single];
                        } 

                        // optimal consumption and labor supply [could be smarter about this interpolation]
                        sim->labor_w[it] = tools::interp_2d(par->grid_Aw,par->grid_K,par->num_A,par->num_K,labor_single_w,Aw_lag,sim->Kw[it]);
                        sim->labor_m[it] = tools::interp_2d(par->grid_Am,par->grid_K,par->num_A,par->num_K,labor_single_m,Am_lag,sim->Km[it]);

                        double resources_w = single::resources_single(sim->labor_w[it],Aw_lag,sim->Kw[it],woman,par); 
                        double resources_m = single::resources_single(sim->labor_m[it],Am_lag,sim->Km[it],man,par); 

                        sim->cons_w[it] = tools::interp_2d(par->grid_Aw,par->grid_K,par->num_A,par->num_K,cons_single_w,Aw_lag,sim->Kw[it]);
                        sim->cons_w[it] = MIN(sim->cons_w[it],resources_w); // cannot borrow. This removes small numerical violations

                        sim->cons_m[it] = tools::interp_2d(par->grid_Am,par->grid_K,par->num_A,par->num_K,cons_single_m,Am_lag,sim->Km[it]);
                        sim->cons_m[it] = MIN(sim->cons_m[it],resources_m); // cannot borrow. This removes small numerical violations
                        

                        // update end-of-period states
                        sim->Aw[it] = resources_w - sim->cons_w[it];
                        sim->Am[it] = resources_m - sim->cons_m[it];

                        if(t<par->T-1){
                            sim->Kw[it1] = utils::K_bar(sim->Kw[it],sim->labor_w[it],par) * sim->draw_Kw[it1];
                            sim->Km[it1] = utils::K_bar(sim->Km[it],sim->labor_m[it],par) * sim->draw_Km[it1];
                        }

                        sim->power[it] = -1.0;

                        // left as nans by not updating them:
                        // sim->power[it1] = nan
                        // sim->love[it] = nan
                        // sim->A[it] = nan
                    }

                } // t
            } // i

        } // pragma

    } // simulate
}