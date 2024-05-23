// functions for bargaining process.
#ifndef MAIN
#define BARGAINING
#include "myheader.cpp"
#endif

namespace bargaining {

    int find_left(double *S, int Nx){

        int left_point {};

        if (S[0] <= S[Nx-1]){
            left_point = tools::binary_search(0, Nx, S, 0.0);
        }
        else{
            left_point = tools::binary_search_over_descending_function(0, Nx, S, 0.0);
        }

        return left_point;
}

    // divorce
    void divorce(int iP, int *power_idx, double *power, index::index_couple_struct *idx_couple, double **list_start_as_couple, double *list_trans_to_single, int num, par_struct *par){
        int idx = idx_couple->idx(iP);
        power_idx[idx] = -1;
        power[idx] = -1.0;

        for (int i = 0; i < num; i++){
            list_start_as_couple[i][idx] = list_trans_to_single[i];
        }
    }
    
    // remain
    void remain(int iP, int *power_idx, double *power, index::index_couple_struct *idx_couple, double **list_start_as_couple, double **list_remain_couple, int num, par_struct *par){
        int idx = idx_couple->idx(iP);
        power_idx[idx] = iP;
        power[idx] = par->grid_power[iP];

        for (int i = 0; i < num; i++){
            list_start_as_couple[i][idx] = list_remain_couple[i][idx];
        }
    }


    // update to indifference point
    void update_to_indifference(int iP, int left_point, int low_point, double power_at_zero, int *power_idx, double *power, index::index_couple_struct *idx_couple, double **list_start_as_couple, double **list_remain_couple, int num, par_struct *par, int sol_idx = -1){
        int idx = idx_couple->idx(iP);
        power_idx[idx] = low_point;
        power[idx] = power_at_zero;

        int delta = idx_couple->idx(left_point+1) - idx_couple->idx(left_point); //difference between the indices of two consecutive values of iP

        // update solution arrays
        if (sol_idx == -1){ // pre-computation not done
            for (int i = 0; i < num; i++){
                list_start_as_couple[i][idx] = tools::interp_1d_index_delta(par->grid_power, par->num_power, list_remain_couple[i], power_at_zero, left_point, delta, idx_couple->idx(0), 1, 0); 
            }
        }
        else{ // pre-computation done - get solution at sol_idx
            for (int i = 0; i < num; i++){
                list_start_as_couple[i][idx] = list_start_as_couple[i][idx_couple->idx(sol_idx)];
            }
        }
    } // end of update_to_indifference


    void limited_commitment(int* power_idx, double* power, double* Sw, double* Sm, index::index_couple_struct* idx_couple, double** list_start_as_couple, double** list_remain_couple, double* list_trans_to_single, int num, par_struct* par){

        // step 0: identify key indicators for each spouse
        // 0a: min and max surplus for each spouse
        double min_w = Sw[0];
        double max_w = Sw[par->num_power-1];
        double min_m = Sm[par->num_power-1];
        double max_m = Sm[0];

        // 0b: check if wife and husband have indifference points
        bool cross_w = (min_w < 0.0) && (max_w > 0.0);
        bool cross_m = (min_m < 0.0) && (max_m > 0.0);

        // 0b: check if wife and husband are always happy
        bool always_happy_w = (min_w > 0.0);
        bool always_happy_m = (min_m > 0.0);

        // 0c: check if wife and husband are never happy
        bool never_happy_w = (max_w < 0.0);
        bool never_happy_m = (max_m < 0.0);

        // step 1: check endpoints
        // 1a. check if all values are consistent with marriage
        if (always_happy_w && always_happy_m){
            for(int iP=0; iP<par->num_power; iP++){
                remain(iP, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par);
            }
        } //case 1a

        // 1b. check if all values are consistent with divorce
        else if (never_happy_w || never_happy_m){

            for (int iP=0; iP<par->num_power; iP++){
                divorce(iP, power_idx, power, idx_couple, list_start_as_couple, list_trans_to_single, num, par);
            }
        } //case 1b

        // 1c. check if husband is always happy, wife has indifference point
        else if (cross_w && always_happy_m){
            // find wife's indifference point
            int left_w = find_left(Sw, par->num_power);
            int Low_w = left_w+1;
            double power_at_zero_w = tools::interp_1d_index(Sw, par->num_power, par->grid_power, 0.0, left_w);

            // update case 1c
            for (int iP=0; iP<par->num_power; iP++){
                if (iP == 0){
                    update_to_indifference(iP, left_w, Low_w, power_at_zero_w, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par, -1);
                }
                else if (iP < Low_w){
                    update_to_indifference(iP, left_w, Low_w, power_at_zero_w, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par, 0);
                }
                else{
                    remain(iP, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par);
                } //if
            } //for
        } //case 1c

        // 1d: check if wife is always happy, husband has indifference point
        else if (cross_m && always_happy_w){
            //find husband's indifference point
            int left_m = find_left(Sm, par->num_power);
            int Low_m = left_m;
            double power_at_zero_m = tools::interp_1d_index(Sm, par->num_power, par->grid_power, 0.0, left_m);

            // update case 1d
            for (int iP=0; iP<par->num_power; iP++){
                if (iP<=Low_m){
                    remain(iP, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par);
                }
                else if (iP==Low_m+1){
                    update_to_indifference(iP, left_m, Low_m, power_at_zero_m, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par, -1);
                }
                else{
                    update_to_indifference(iP, left_m, Low_m, power_at_zero_m, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par, Low_m+1);
                } //if
            } //for
        } //case 1d

        // 1e: Both have indifference points
        else {
            //find indifference points
            int left_w = find_left(Sw, par->num_power);
            int Low_w = left_w+1;
            double power_at_zero_w = tools::interp_1d_index(Sw, par->num_power, par->grid_power, 0.0, left_w);

            int left_m = find_left(Sm, par->num_power);
            int Low_m = left_m;         
            double power_at_zero_m = tools::interp_1d_index(Sm, par->num_power, par->grid_power, 0.0, left_m);

            // update case 1e
            // no room for bargaining
            if (power_at_zero_w>power_at_zero_m) {
                for (int iP=0; iP<par->num_power; iP++){
                    divorce(iP, power_idx, power, idx_couple, list_start_as_couple, list_trans_to_single, num, par);
                }
            }
            //bargaining
            else {
                for (int iP=0; iP<par->num_power; iP++){
                    if (iP==0){ //update to woman's indifference point
                        update_to_indifference(iP, left_w, Low_w, power_at_zero_w, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par, -1);
                    }
                    else if (iP<Low_w){ //re-use pre-computed values
                        update_to_indifference(iP, left_w, Low_w, power_at_zero_w, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par, 0);
                    }
                    else if (iP>=Low_w && iP <= Low_m) { //no change between Low_w and Low_m
                        remain(iP, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par);
                    }
                    else if (iP == Low_m+1) { //update to man's indifference point
                        update_to_indifference(iP, left_m, Low_m, power_at_zero_m, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par, -1);
                    }
                    else { // re-use precomputed values
                        update_to_indifference(iP, left_m, Low_m, power_at_zero_m, power_idx, power, idx_couple, list_start_as_couple, list_remain_couple, num, par, Low_m+1);
                    } //if (indifference points)
                }//for
            } //if (bargaining)
        } //case 1e
    } //end of check_participation_constraints

    void full_commitment(int* power_idx, double* power, double* Sw, double* Sm, index::index_couple_struct* idx_couple, double** list_start_as_couple, double** list_remain_couple, double* list_trans_to_single, int num, par_struct* par){
        
        for (int iP=0; iP<par->num_power; iP++){
            int idx = idx_couple->idx(iP);

            if((Sw[iP]<0.0)|(Sm[iP]<0.0)){
            //if((Sw[iP]+Sm[iP]<0.0)){ // transferable utility
                for(int i=0;i<num;i++){
                    list_start_as_couple[i][idx] = list_trans_to_single[i];
                }
                power_idx[idx] = -1;
                power[idx] = -1.0;

            } else {
                
                for(int i=0;i<num;i++){
                    list_start_as_couple[i][idx] = list_remain_couple[i][idx];
                }

                power_idx[idx] = iP;
                power[idx] = par->grid_power[iP];

            }
        }

    } // FULL COMMITMENT

    // NASH BARGAINING
    typedef struct {        
        par_struct *par;   
        sol_struct *sol;                                    
        double (*surplus_func)(double,index::state_couple_struct*,index::state_single_struct*,int,par_struct*,sol_struct*); //surplus func as function of power and state
        index::state_couple_struct *state_couple;                                       // state - tbc 
        index::state_single_struct *state_single_w;                                       // state - tbc   
        index::state_single_struct *state_single_m;                                       // state - tbc                             
    } nash_solver_struct;


    double surplus_func(double power, index::state_couple_struct* state_couple, index::state_single_struct* state_single, int gender, par_struct* par, sol_struct* sol){
        // unpack
        int t = state_couple->t;
        double Kw = state_couple->Kw;
        double Km = state_couple->Km;
        double love = state_couple->love;
        double A_couple = state_couple->A; 
        double A_single = state_single->A;

        // gender specific arrays
        double* V_couple_to_single = sol->Vw_trans_single;
        double* V_couple_to_couple = sol->Vw_remain_couple;
        double* grid_A_single = par->grid_Aw;
        if (gender == man){
            V_couple_to_single = sol->Vm_trans_single;
            V_couple_to_couple = sol->Vm_remain_couple;
            grid_A_single = par->grid_Am;
        }
        
        // Get indices (could be faster if indexes passed in solution)
        int iZw = state_couple->iZw;
        int iZm = state_couple->iZm;

        int iL = tools::binary_search(0, par->num_love, par->grid_love, love);
        int iKw = tools::binary_search(0, par->num_K, par->grid_K, Kw);
        int iKm = tools::binary_search(0, par->num_K, par->grid_K, Km);
        int iA_couple = tools::binary_search(0, par->num_A, par->grid_A, A_couple);
        int iA_single = tools::binary_search(0, par->num_A, grid_A_single, A_single);
        int iP = tools::binary_search(0, par->num_power, par->grid_power, power);

        // gender specific indices
        int iZ_single = iZw;
        int iK_single = iKw;
        double K_single = Kw;
        if (gender == man){
            iZ_single = iZm;
            iK_single = iKm;
            K_single = Km;
        }

        //interpolate V_couple_to_single
        int idx_single = index::single(t,iZ_single,0,0,par);
        double V_single = tools::_interp_2d(grid_A_single,par->grid_K, 
                                            par->num_A, par->num_K,
                                            &V_couple_to_single[idx_single], 
                                            A_single,K_single, 
                                            iA_single,iK_single); 

        // interpolate couple V_couple_to_couple  
        int idx_couple = index::couple(t,iZw,iZm,0,0,0,0,0,par); //couple(t,iZw,iZm,iP,iL,iA,iKw,iKm,par) index::couple(t,0,0,0,par);
        double V_couple = tools::_interp_5d(par->grid_power, par->grid_love, par->grid_A,par->grid_K, par->grid_K, 
                                       par->num_power, par->num_love, par->num_A, par->num_K, par->num_K,
                                       &V_couple_to_couple[idx_couple], power, love, A_couple, Kw, Km,
                                       iP, iL, iA_couple,iKw,iKm);

        // surplus
        return V_couple - V_single;
    }

 
    // compute negative nash surplus for given power + nash_struct
    double objfunc_nash_bargain(unsigned n, const double *x, double *grad, void* solver_data_in){
        // unpack
        nash_solver_struct* solver_data = (nash_solver_struct*) solver_data_in; 
        par_struct* par = solver_data->par;
        sol_struct* sol = solver_data->sol;
        double (*surplus_func)(double,index::state_couple_struct*, index::state_single_struct*,int,par_struct*,sol_struct*) = solver_data->surplus_func; //TODO: take continuous states as input as generically as possible

        // calculate individual surpluses
        double Sw_x = surplus_func(x[0],solver_data->state_couple, solver_data->state_single_w, woman, par, sol);
        double Sm_x = surplus_func(x[0],solver_data->state_couple, solver_data->state_single_m, man, par, sol);

        // make sure surpluses are positive
        double penalty = 0.0;
        if(Sw_x<0.0){ 
            penalty += 1000.0*Sw_x;
            Sw_x = 0.0; 
        } 
        if (Sm_x<0.0){ 
            penalty += 1000.0*Sm_x;
            Sm_x = 0.0;
        }

        return -(Sw_x*Sm_x) - penalty; 

    }

    
    double nash_bargain(nash_solver_struct* nash_struct){
        // for a given couple idx, find the bargaining weight

        // unpack
        par_struct* par = nash_struct->par;
        sol_struct* sol = nash_struct->sol;

        // set up solver
        int const dim = 1;
        auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim);
        nlopt_set_min_objective(opt, objfunc_nash_bargain, nash_struct);

        // set bounds
        double lb[dim], ub[dim];
        lb[0] = par->grid_power[0];
        ub[0] = par->grid_power[par->num_power-1];
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        //optimize
        double minf = 0.0;
        double mu[dim];
        mu[0] = 0.5;
        nlopt_optimize(opt, mu, &minf);
        nlopt_destroy(opt);

        // check surplus is positive
        double (*surplus_func)(double,index::state_couple_struct*, index::state_single_struct*,int,par_struct*,sol_struct*) = nash_struct->surplus_func; //TODO: take continuous states as input as generically as possible
        index::state_couple_struct* state_couple = nash_struct->state_couple;
        index::state_single_struct* state_single_w = nash_struct->state_single_w;
        index::state_single_struct* state_single_m = nash_struct->state_single_m;
        double Sw = surplus_func(mu[0], state_couple, state_single_w, woman, par, sol);
        double Sm = surplus_func(mu[0], state_couple, state_single_m, man, par, sol);
        if ((Sw<0.0) | (Sm<0.0)){
            mu[0] = -1.0;
        }

        return mu[0];
    }

    double calc_bargaining_weight(int t, double love, double Aw,double Am, double Kw, double Km, int iZw, int iZm, sol_struct* sol, par_struct* par){
        // state structs
        index::state_couple_struct* state_couple = new index::state_couple_struct;
        index::state_single_struct* state_single_w = new index::state_single_struct;
        index::state_single_struct* state_single_m = new index::state_single_struct;

        // couple
        state_couple->t = t;
        state_couple->love = love;
        state_couple->A = Aw+Am;
        state_couple->Kw = Kw;
        state_couple->Km = Km;
        state_couple->iZw = iZw;
        state_couple->iZm = iZm;

        // single woman
        state_single_w->t = t;
        state_single_w->A = Aw;

        // single man
        state_single_m->t = t;
        state_single_m->A = Am;

        //solver input
        nash_solver_struct* nash_struct = new nash_solver_struct;
        nash_struct->surplus_func = surplus_func;
        nash_struct->state_couple = state_couple;
        nash_struct->state_single_w = state_single_w;
        nash_struct->state_single_m = state_single_m;
        nash_struct->sol = sol;
        nash_struct->par = par;

        // solve
        double power =  nash_bargain(nash_struct);

        delete state_couple;
        delete state_single_w;
        delete state_single_m;
        delete nash_struct;

        return power;
    }


} // namespace bargaining