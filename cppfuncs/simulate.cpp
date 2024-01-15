
// functions for solving model for singles.
#ifndef MAIN
#define SIMULATE
#include "myheader.cpp"
#endif

namespace sim {

    int update_power_index(int t, int power_idx_lag, double love,double A_lag,double Aw_lag,double Am_lag,double Kw, double Km,sim_struct* sim, sol_struct* sol, par_struct* par){
        
        // value of transitioning into singlehood
        int idx_single = index::single(t,0,0,par);
        double Vw_single = tools::interp_2d(par->grid_Aw,par->grid_K,par->num_A,par->num_K,&sol->Vw_trans_single[idx_single],Aw_lag,Kw);
        double Vm_single = tools::interp_2d(par->grid_Am,par->grid_K,par->num_A,par->num_K,&sol->Vm_trans_single[idx_single],Am_lag,Km);

        // value of remaining a couple. [could be smarter about this interpolation]
        int idx_sol = index::couple(t,power_idx_lag,0,0,0,0,par); 
        double Vw_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx_sol], love,A_lag,Kw,Km);
        double Vm_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vm_remain_couple[idx_sol], love,A_lag,Kw,Km);

        // check participation constraints
        int power_idx = -1; // initialize as divorce
        if ((Vw_couple>=Vw_single) & (Vm_couple>=Vm_single)){
            power_idx = power_idx_lag;

        } else {

            if ((Vm_couple>=Vm_single)){ // woman wants to leave

                // for (int iP=power_idx_lag+1; iP<(par->num_power-power_idx_lag); iP++){ // increase power of women I think this loop is wrong in the guide!! should go all the way to num_power
                    // int idx = index::index4(t,iP,0,0,par->T,par->num_power,par->num_love,par->num_A); 
                    // tools::interp_2d_2out(par->grid_love,par->grid_A,par->num_love,par->num_A,&sol->Vw_remain_couple[idx],&sol->Vm_remain_couple[idx],love,A_lag, &Vw_couple, &Vm_couple);
                    
                for (int iP=power_idx_lag+1; iP<par->num_power; iP++){ 
                    int idx = index::couple(t,iP,0,0,0,0,par);  // could agin be smarter with the interpolation
                    Vw_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx], love,A_lag,Kw,Km);
                    Vm_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vm_remain_couple[idx], love,A_lag,Kw,Km);

                    // check participation constraint
                    double Sw = couple::calc_marital_surplus(Vw_couple,Vw_single,par);
                    double Sm = couple::calc_marital_surplus(Vm_couple,Vm_single,par);

                    // update index if a solution is found
                    if((Sw>=0.0)&(Sm>=0.0)){
                        power_idx = iP; 
                        break;
                    }
                    
                }

            } else { // man wants to leave

                for (int iP=power_idx_lag-1; iP>=0; iP--){ // increase power of men
                    // int idx = index::index4(t,iP,0,0,par->T,par->num_power,par->num_love,par->num_A); 
                    // tools::interp_2d_2out(par->grid_love,par->grid_A,par->num_love,par->num_A,&sol->Vw_remain_couple[idx],&sol->Vm_remain_couple[idx],love,A_lag, &Vw_couple, &Vm_couple);
                    int idx = index::couple(t,iP,0,0,0,0,par);  // could agin be smarter with the interpolation
                    Vw_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vw_remain_couple[idx], love,A_lag,Kw,Km);
                    Vm_couple = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->Vm_remain_couple[idx], love,A_lag,Kw,Km);

                    // check participation constraint
                    double Sw = couple::calc_marital_surplus(Vw_couple,Vw_single,par);
                    double Sm = couple::calc_marital_surplus(Vm_couple,Vm_single,par);

                    // update index if a solution is found
                    if((Sw>=0.0)&(Sm>=0.0)){
                        power_idx = iP; 
                        break;
                    }
                    
                }

            }
        }

        return power_idx;
    } // update_power_index


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
                    int power_idx_lag = sim->init_power_idx[i];

                    // state variables
                    if (t==0){
                        
                        // in other periods these states will be updated in the end of the period below
                        sim->love[it] = sim->init_love[i];
                        sim->Kw[it] = sim->init_Kw[i];
                        sim->Km[it] = sim->init_Km[i];

                    } else {
                        A_lag = sim->A[it_1];
                        Aw_lag = sim->Aw[it_1];
                        Am_lag = sim->Am[it_1];
                        couple_lag = sim->couple[it_1];
                        power_idx_lag = sim->power_idx[it_1];
                        
                    } 

                    
                    // first check if they want to remain together and what the bargaining power will be if they do.
                    int power_idx;
                    if (couple_lag) {

                        power_idx = update_power_index(t,power_idx_lag,sim->love[it],A_lag,Aw_lag,Am_lag,sim->Kw[it],sim->Km[it],sim,sol,par);

                        if (power_idx < 0) { // divorce is coded as -1
                            sim->couple[it] = false;

                        } else {
                            sim->couple[it] = true;
                        }

                    } else { // remain single
                        sim->couple[it] = false;
                    }

                    // update behavior
                    if (sim->couple[it]){
                        int idx_sol = index::couple(t,power_idx,0,0,0,0,par);

                        // optimal labor supply and consumption [could be smarter about this interpolation]
                        sim->labor_w[it] = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->labor_w_remain_couple[idx_sol], sim->love[it],A_lag,sim->Kw[it],sim->Km[it]);
                        sim->labor_m[it] = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->labor_m_remain_couple[idx_sol], sim->love[it],A_lag,sim->Kw[it],sim->Km[it]);
                        
                        double resources = couple::resources(sim->labor_w[it],sim->labor_m[it],A_lag,sim->Kw[it],sim->Kw[it],par); 
                        double cons = tools::interp_4d(par->grid_love,par->grid_A,par->grid_K,par->grid_K ,par->num_love,par->num_A,par->num_K,par->num_K, &sol->cons_w_remain_couple[idx_sol], sim->love[it],A_lag,sim->Kw[it],sim->Kw[it]); // same for men and women in remain couple
                        cons = MIN(cons,resources); // cannot borrow. This removes small numerical violations
                        sim->cons_w[it] = cons;
                        sim->cons_m[it] = cons;

                        // update end-of-period states
                        sim->A[it] = resources - cons;
                        if(t<par->T-1){
                            sim->love[it1] = sim->love[it] + par->sigma_love * sim->draw_love[it1];
                            sim->Kw[it1] = utils::K_bar(sim->Kw[it],sim->labor_w[it],par) * sim->draw_Kw[it1];
                            sim->Km[it1] = utils::K_bar(sim->Km[it],sim->labor_m[it],par) * sim->draw_Km[it1];
                        }

                        // in case of divorce
                        sim->Aw[it] = par->div_A_share * sim->A[it];
                        sim->Am[it] = (1.0-par->div_A_share) * sim->A[it];

                        sim->power_idx[it] = power_idx;
                        sim->power[it] = par->grid_power[power_idx];

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

                        sim->power_idx[it] = -1;

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