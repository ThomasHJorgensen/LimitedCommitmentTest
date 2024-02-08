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

    typedef struct {  
        par_struct* par;   
        double* Sw;
        double* Sm;          
    } solver_struct_nash;

    double obj_nash(unsigned n, const double *x, double *grad, void *solver_data_in){
        // unpack
        double power = x[0];

        solver_struct_nash *solver_data = (solver_struct_nash *) solver_data_in;
        double* Sw = solver_data->Sw;
        double* Sm = solver_data->Sm;
        par_struct* par = solver_data->par;

        // penalty and clip
        double penalty = 0.0;
        if(power<0.0){ 
            penalty += 1000.0*power*power;
            power = 0.0; 
        } else if (power>1.0){ 
            penalty += 1000.0*(power-1.0)*(power-1.0);
            power = 1.0;
        }

        // interpolate surplus
        int j = tools::binary_search(0, par->num_power, par->grid_power, power);
        double Sw_interp = tools::interp_1d_index(par->grid_power, par->num_power, Sw, power, j);
        double Sm_interp = tools::interp_1d_index(par->grid_power, par->num_power, Sm, power, j);

        if(Sw_interp<0.0){ 
            penalty += 1000.0*Sw_interp*Sw_interp;
            Sw_interp = 0.0; 
        } else if (Sm_interp<0.0){ 
            penalty += 1000.0*Sm_interp*Sm_interp;
            Sm_interp = 0.0;
        }

        // Nash objective function
        double obj = sqrt(Sw_interp) * sqrt(Sm_interp);

        // return negative for minimization
        return - obj + penalty;

    }

    void nash(int* power_idx, double* power, double* Sw, double* Sm, index::index_couple_struct* idx_couple, double** list_start_as_couple, double** list_remain_couple, double* list_trans_to_single, int num, par_struct* par){
        // find (discrete) max TODO: think about continuous max (interpolation)
        double obj_max = -1.0e10;
        int iP_max = -1;
        for (int iP=0; iP<par->num_power; iP++){

            if((Sw[iP]>0.0) & (Sm[iP]>0.0)){
                
                double obj_now = sqrt(Sw[iP]) * sqrt(Sm[iP]);
                
                if(obj_now>obj_max){
                    obj_max = obj_now;
                    iP_max = iP;
                }
            }
        }

        // update solution
        if(iP_max>-1){
            bool do_cont = true;

            if(do_cont){
                // continuous optimization
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

                solver_struct_nash* solver_data = new solver_struct_nash;
                solver_data->par = par;
                solver_data->Sw = Sw;
                solver_data->Sm = Sm;
                nlopt_set_min_objective(opt, obj_nash, solver_data); 

                // optimize
                y[0] = par->grid_power[iP_max];
                nlopt_optimize(opt, y, &minf); 
                double power_est = y[0];

                // destroy optimizer
                nlopt_destroy(opt);

                // interpolate solutions onto grids
                int left_point = tools::binary_search(0, par->num_power, par->grid_power, power_est);
                int delta = idx_couple->idx(1) - idx_couple->idx(0);
                for (int iP=0; iP<par->num_power; iP++){
                    int idx = idx_couple->idx(iP);
                    for (int i=0;i<num;i++){
                        list_start_as_couple[i][idx] = tools::interp_1d_index_delta(par->grid_power, par->num_power, list_remain_couple[i], power_est, left_point, delta, idx_couple->idx(0)); 
                    }
                    power[idx] = power_est;
                    power_idx[idx] = iP_max; // close to the optimal
                }

            } else {
                // discrete optimization
                int idx_max = idx_couple->idx(iP_max);
                for (int iP=0; iP<par->num_power; iP++){
                    int idx = idx_couple->idx(iP);
                    for (int i=0;i<num;i++){
                        list_start_as_couple[i][idx] = list_remain_couple[i][idx_max];
                    }

                    power_idx[idx] = iP_max;
                    power[idx] = par->grid_power[iP_max];
                }
            }

        } else {

            // divorce
            for (int iP=0; iP<par->num_power; iP++){
                int idx = idx_couple->idx(iP);
                for (int i=0;i<num;i++){
                    list_start_as_couple[i][idx] = list_trans_to_single[i];
                }

                power_idx[idx] = -1;
                power[idx] = -1.0;

            }
        }

    } // NASH

    void full_commitment(int* power_idx, double* power, double* Sw, double* Sm, index::index_couple_struct* idx_couple, double** list_start_as_couple, double** list_remain_couple, double* list_trans_to_single, int num, par_struct* par){
        
        for (int iP=0; iP<par->num_power; iP++){
            int idx = idx_couple->idx(iP);

            // if((Sw[iP]<0.0)|(Sm[iP]<0.0)){
            if((Sw[iP]+Sm[iP]<0.0)){ // transferable utility
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

} // namespace bargaining