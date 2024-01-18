
// functions for solving model for singles.
#ifndef MAIN
#define SIMULATE
#include "myheader.h"
#endif

namespace sim {

    double limited_commitment(int t,int idx_sol, double power_lag,double Vw_couple,double Vm_couple,double Vw_single,double Vm_single, double love,double A_lag,double Aw_lag,double Am_lag,double Kw, double Km,sim_struct* sim, sol_struct* sol, par_struct* par,int bargaining){
        // check participation constraints
        double power = -1.0; // initialize as divorce

        if ((Vw_couple>=Vw_single) & (Vm_couple>=Vm_single)){
            power = power_lag;

        } else {

            if ((Vm_couple>=Vm_single)){ // woman wants to leave

                // find relevant values of remaining a couple at all other states than power 
                double* Vw_power = new double[par->num_power];
                int j_love = tools::binary_search(0,par->num_love,par->grid_love,love); 
                int j_A = tools::binary_search(0,par->num_A,par->grid_A,A_lag);
                int j_Kw = tools::binary_search(0,par->num_K,par->grid_K,Kw);
                int j_Km = tools::binary_search(0,par->num_K,par->grid_K,Km);
                for (int iP=0; iP<par->num_power; iP++){ 
                    int idx = index::couple(t,iP,0,0,0,0,par); 
                    Vw_power[iP] = tools::_interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K, par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx],love,A_lag,j_love,j_A,j_Kw,j_Km);
                }

                // interpolate the power based on the value of single to find indifference-point. (flip the axis so to speak)
                power = tools::interp_1d(Vw_power, par->num_power, par->grid_power, Vw_single);
                delete Vw_power;

                // calculate value for man at this level of power
                double Vm_power = tools::interp_5d(par->grid_power,par->grid_love,par->grid_A,par->grid_K,par->grid_K, par->num_power,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vm_remain_couple[idx_sol], power,love,A_lag,Kw,Km);
                double Sm = couple::calc_marital_surplus(Vm_power,Vm_single,par);
                
                // check if he is happy
                if(Sm<0.0){
                    power = -1.0;  // he is not -> divorce
                }


            } else { // man wants to leave

                // find relevant values of remaining a couple at all other states than power 
                double* Vm_power = new double[par->num_power];
                int j_love = tools::binary_search(0,par->num_love,par->grid_love,love); 
                int j_A = tools::binary_search(0,par->num_A,par->grid_A,A_lag);
                int j_Kw = tools::binary_search(0,par->num_K,par->grid_K,Kw);
                int j_Km = tools::binary_search(0,par->num_K,par->grid_K,Km);
                for (int iP=0; iP<par->num_power; iP++){ 
                    int idx = index::couple(t,iP,0,0,0,0,par); 
                    Vm_power[iP] = tools::_interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K, par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vm_remain_couple[idx],love,A_lag,j_love,j_A,j_Kw,j_Km);
                }

                // interpolate the power based on the value of single to find indifference-point. (flip the axis so to speak)
                power = tools::interp_1d(Vm_power, par->num_power, par->grid_power, Vm_single);
                delete Vm_power;

                // calculate value for man at this level of power
                double Vw_power = tools::interp_5d(par->grid_power,par->grid_love,par->grid_A,par->grid_K,par->grid_K, par->num_power,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx_sol], power,love,A_lag,Kw,Km);
                double Sw = couple::calc_marital_surplus(Vw_power,Vw_single,par);
                
                // check if she is happy
                if(Sw<0.0){
                    power = -1.0;  // she is not -> divorce
                }

            }
        }

        return power;
    }

    double update_power(int t, int power_lag, double love,double A_lag,double Aw_lag,double Am_lag,double Kw, double Km,sim_struct* sim, sol_struct* sol, par_struct* par,int bargaining){
        
        
        // value of transitioning into singlehood
        int idx_single = index::single(t,0,0,par);
        double Vw_single = tools::interp_2d(par->grid_Aw,par->grid_K,par->num_A,par->num_K,&sol->Vw_trans_single[idx_single],Aw_lag,Kw);
        double Vm_single = tools::interp_2d(par->grid_Am,par->grid_K,par->num_A,par->num_K,&sol->Vm_trans_single[idx_single],Am_lag,Km);

        // value of remaining a couple with current power. [could be smarter about this interpolation]
        int idx_sol = index::couple(t,0,0,0,0,0,par); 
        double Vw_couple,Vm_couple;
        tools::interp_5d_2out(&Vw_couple,&Vm_couple,par->grid_power,par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx_sol],&sol->Vw_remain_couple[idx_sol], power_lag,love,A_lag,Kw,Km);
        // double Vw_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx_sol], love,A_lag,Kw,Km);
        // double Vm_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vm_remain_couple[idx_sol], love,A_lag,Kw,Km);

        
        if(bargaining==0){ // no bargaining: full commitment
            if((Vw_couple<Vw_single) | (Vm_couple<Vm_single)){
                return -1.0;
            } else {
                return power_lag;
            }

        } else if(bargaining==2){ // NASH
            // TODO: re-use? or build new. latter easiest in some sense.

            // // find (discrete) max
            // double obj_max = -1.0e10;
            // int iP_max = -1;
            // for (int iP=0; iP<par->num_power; iP++){
            //     // value of remaining a couple. [could be smarter about this interpolation]
            //     int idx_sol = index::couple(t,iP,0,0,0,0,par); 
            //     double Vw_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx_sol], love,A_lag,Kw,Km);
            //     double Vm_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vm_remain_couple[idx_sol], love,A_lag,Kw,Km);

            //     // marital surplus
            //     double Sw = couple::calc_marital_surplus(Vw_couple,Vw_single,par);
            //     double Sm = couple::calc_marital_surplus(Vm_couple,Vm_single,par);

            //     if((Sw>0.0) & (Sm>0.0)){
                
            //         double obj_now = sqrt(Sw) * sqrt(Sm);
                    
            //         if(obj_now>obj_max){
            //             obj_max = obj_now;
            //             iP_max = iP;
            //         }
            //     }

            // }

            // // return relevant index
            // return iP_max;

        } else { // limited commitment

            return limited_commitment(t,idx_sol,power_lag,Vw_couple,Vm_couple,Vw_single,Vm_single, love,A_lag,Aw_lag,Am_lag,Kw,Km,sim,sol,par,bargaining);
            
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

                        power = update_power(t,power_lag,sim->love[it],A_lag,Aw_lag,Am_lag,sim->Kw[it],sim->Km[it],sim,sol,par,bargaining);

                        if (power < 0) { // divorce is coded as -1
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

                        // optimal labor supply and consumption [could be smarter about this interpolation]
                        tools::interp_5d_2out(&sim->labor_w[it],&sim->labor_m[it], par->grid_power,par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx_sol],&sol->Vw_remain_couple[idx_sol], power,sim->love[it],A_lag,sim->Kw[it],sim->Km[it]);
                        // sim->labor_w[it] = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->labor_w_remain_couple[idx_sol], sim->love[it],A_lag,sim->Kw[it],sim->Km[it]);
                        // sim->labor_m[it] = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->labor_m_remain_couple[idx_sol], sim->love[it],A_lag,sim->Kw[it],sim->Km[it]);
                        
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
                        int idx_sol_single = index::single(t,0,0,par); //index::index2(t,0,par->T,par->num_A);
                        double *labor_single_w = &sol->labor_w_trans_single[idx_sol_single];
                        double *cons_single_w = &sol->cons_w_trans_single[idx_sol_single];
                        double *labor_single_m = &sol->labor_m_trans_single[idx_sol_single];
                        double *cons_single_m = &sol->cons_m_trans_single[idx_sol_single];
                        if (power_idx_lag<0){
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