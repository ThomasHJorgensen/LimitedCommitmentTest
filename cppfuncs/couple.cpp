
// functions for solving model for couples.
#ifndef MAIN
#define COUPLE
#include "myheader.cpp"
#endif

namespace couple {
    
    typedef struct {                
        int iP;  
        double love;
        double A;
        double Kw;
        double Km;           
        double *Vw;    
        double *Vm;     
        double *EVw_next;    
        double *EVm_next;    

        int t;

        double labor_w;
        double labor_m;
        double cons;

        sol_struct *sol;
        par_struct *par;

        double *lower;
        double *upper;

    } solver_couple_struct;

    double calc_marital_surplus(double V_remain_couple,double V_trans_single,par_struct* par){
        return V_remain_couple - V_trans_single;
    }


    EXPORT double resources(double labor_w, double labor_m,double A,double Kw, double Km,par_struct* par) {
        double income_w = labor_w * utils::wage_func(Kw,woman,par);
        double income_m = labor_m * utils::wage_func(Km,man,par);
        return par->R*A + income_w + income_m;
    }

    // pre-compute
    double precompute_EV(int t_next,sol_struct* sol,par_struct* par){
        // precompute expected continuation values for men and women
        // over power, love, Abar, Kwbar, Kmbar. 
        
        // loop through end-of-period states
        #pragma omp parallel num_threads(par->threads)
        {
        #pragma omp for
        for (int iP=0; iP<par->num_power; iP++){
            for (int iL=0; iL<par->num_love; iL++){
                double love = par->grid_love[iL];

                int jA = 0;
                for (int iA=0; iA<par->num_A_pd; iA++){
                    double A_next = par->grid_A_pd[iA];
                    jA = tools::binary_search(jA,par->num_A,par->grid_A,A_next);

                    for (int iKw=0; iKw<par->num_K_pd; iKw++){
                        double Kbar_w = par->grid_K_pd[iKw];
                        for (int iKm=0; iKm<par->num_K_pd; iKm++){
                            double Kbar_m = par->grid_K_pd[iKm];
                            
                            // next-period values
                            int idx_next = index::couple(t_next,iP,0,0,0,0,par);
                            double* Vw_next = &sol->Vw_remain_couple[idx_next]; 
                            double* Vm_next = &sol->Vm_remain_couple[idx_next]; 

                            // loop through shocks
                            double EVw_plus = 0.0;
                            double EVm_plus = 0.0;
                            
                            int jKw = 0;
                            for (int iKw_next=0;iKw_next<par->num_shock_K;iKw_next++){
                                double Kw_next = Kbar_w*par->grid_shock_K[iKw_next];
                                jKw = tools::binary_search(jKw,par->num_K,par->grid_K,Kw_next);
                                
                                int jKm = 0;
                                for (int iKm_next=0;iKm_next<par->num_shock_K;iKm_next++){
                                    double Km_next = Kbar_m*par->grid_shock_K[iKm_next];
                                    jKm = tools::binary_search(jKm,par->num_K,par->grid_K,Km_next);
                                    
                                    int jL = 0;
                                    for (int iL_next = 0; iL_next < par->num_shock_love; iL_next++) {
                                        double love_next = love + par->grid_shock_love[iL_next];
                                        jL = tools::binary_search(jL,par->num_love,par->grid_love,love_next);
                                        
                                        double Vw_interp, Vm_interp;
                                        tools::_interp_4d_2out(&Vw_interp,&Vm_interp, par->grid_love,par->grid_A,par->grid_K,par->grid_K,par->num_love,par->num_A,par->num_K,par->num_K, Vw_next,Vm_next, love_next,A_next,Kw_next,Km_next, jL,jA,jKw,jKm);
                                        
                                        double weight = par->grid_weight_love[iL_next] * par->grid_weight_K[iKw_next] * par->grid_weight_K[iKm_next];
                                        EVw_plus += weight * Vw_interp;
                                        EVm_plus += weight * Vm_interp;

                                    }
                                }
                            }

                            // store in solution
                            int idx = index::precomp(iP,iL,iA,iKw,iKm,par);
                            sol->EVw_pd[idx] = par->beta*EVw_plus;
                            sol->EVm_pd[idx] = par->beta*EVm_plus;

                        } // Kmbar
                    } // Kwbar
                } // Abar
            } // love
        } // power

        } // pragma
    
    }

    double value_of_choice(double* Vw,double* Vm,double cons,double labor_w,double labor_m,double power,double love,double A,double Kw,double Km, int t,double* EVw_next,double* EVm_next,par_struct* par){
        // current utility flow
        Vw[0] = utils::util(cons,labor_w,woman,par) + love;
        Vm[0] = utils::util(cons,labor_m,man,par) + love;

        // add continuation value 
        if(t<par->T-1){
            double A_next = resources(labor_w,labor_m,A,Kw,Km,par) - cons ;
            double Kbar_w = utils::K_bar(Kw,labor_w,par);
            double Kbar_m = utils::K_bar(Km,labor_m,par);

            // interpolate next-period expected value (pre-computed) wrt. Anext,Kwbar,Kmbar (love is on grid)
            double EVw_plus,EVm_plus;
            tools::interp_3d_2out(&EVw_plus,&EVm_plus, par->grid_A_pd,par->grid_K_pd,par->grid_K_pd, par->num_A_pd,par->num_K_pd,par->num_K_pd, EVw_next,EVm_next ,A_next,Kbar_w,Kbar_m);

            Vw[0] += EVw_plus;
            Vm[0] += EVm_plus;

        }

        // return
        return power*Vw[0] + (1.0-power)*Vm[0];
    }

    double objfunc_cons(unsigned n, const double *x, double *grad, void *solver_data_in){
        // unpack
        solver_couple_struct *solver_data = (solver_couple_struct *) solver_data_in;

        double cons = x[0];

        int iP = solver_data->iP;
        double love = solver_data->love;
        double A = solver_data->A;
        double Kw = solver_data->Kw;
        double Km = solver_data->Km;

        // sol_struct *sol = solver_data->sol;
        par_struct *par = solver_data->par;

        double* Vw = solver_data->Vw;
        double* Vm = solver_data->Vm;

        int t = solver_data->t;

        double labor_w = solver_data->labor_w;
        double labor_m = solver_data->labor_m;

        double power = par->grid_power[iP];

        // penalty and clip
        double penalty = 0.0;
        double saving = resources(labor_w,labor_m,A,Kw,Km,par) - cons;
        if(saving<0.0){ // budget constraint: no borrowing
            penalty += 1000.0*saving*saving;
            cons -= saving; 
        }
        double low_cons = 1.0e-6;
        if (cons <low_cons) {
            penalty += 1000.0*(low_cons-cons)*(low_cons-cons);
            cons = low_cons;
        } 

        // return negative value of choice
        return - value_of_choice(Vw,Vm,cons,labor_w,labor_m,power,love,A,Kw,Km,t,solver_data->EVw_next,solver_data->EVm_next,par) + penalty;

    }

    double objfunc_labor(unsigned n, const double *x, double *grad, void *solver_data_in){
        // unpack
        solver_couple_struct *solver_data = (solver_couple_struct *) solver_data_in;

        double labor_w = x[0];
        double labor_m = x[1];

        par_struct *par = solver_data->par;

        int iP = solver_data->iP;
        double love = solver_data->love;
        double A = solver_data->A;
        double Kw = solver_data->Kw;
        double Km = solver_data->Km;
        double power = par->grid_power[iP];

        double* lower = solver_data->lower;
        double* upper = solver_data->upper;

        double* Vw = solver_data->Vw;
        double* Vm = solver_data->Vm;

        int t = solver_data->t;

        solver_data->labor_w = labor_w;
        solver_data->labor_m = labor_m;

        // penalty and clip
        double penalty = 0.0;
        if (labor_w < lower[0]) {
            penalty += 1000.0*(lower[0]-labor_w)*(lower[0]-labor_w);
            labor_w = lower[0];
        } else if (labor_w > upper[0]) {
            penalty += 1000.0*(upper[0]-labor_w)*(upper[0]-labor_w);
            labor_w = upper[0];
        }
        if (labor_m < lower[1]) {
            penalty += 1000.0*(lower[1]-labor_m)*(lower[1]-labor_m);
            labor_m = lower[1];
        } else if (labor_m > upper[1]) {
            penalty += 1000.0*(upper[1]-labor_m)*(upper[1]-labor_m);
            labor_m = upper[1];
        } 

        // solve for optimal consumption at this level of labor supply
        double minf=0.0;
        if (t<(par->T-1)){
            int const dim = 1;
            double lb[dim],ub[dim],y[dim];
            
            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT

            // bounds
            lb[0] = 1.0e-6;
            ub[0] = resources(labor_w,labor_m,A,Kw,Km,par)-1.0e-6;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            nlopt_set_ftol_rel(opt,1.0e-5);
            nlopt_set_xtol_rel(opt,1.0e-5);

            
            nlopt_set_min_objective(opt, objfunc_cons, solver_data); 

            // optimize
            y[0] = MIN(ub[0],solver_data->cons); // use last found consumption as starting guess
            nlopt_optimize(opt, y, &minf); 
            solver_data->cons = y[0];

            // destroy optimizer
            nlopt_destroy(opt);

        } else {
            // consume all resources in last period
            solver_data->cons = resources(labor_w,labor_m,A,Kw,Km,par);
            minf = - value_of_choice(Vw,Vm,solver_data->cons,labor_w,labor_m,power,love,A,Kw,Km,t,solver_data->EVw_next,solver_data->EVm_next,par);

        }

        // return objective function
        return minf + penalty;

    }

    void solve_remain(int t, int iP, int iL,int iA, int iKw, int iKm, double* EVw_next, double* EVm_next, sol_struct* sol, par_struct* par){
        
        int idx = index::couple(t,iP,iL,iA,iKw,iKm,par);

        double love = par->grid_love[iL];
        double A = par->grid_A[iA];
        double Kw = par->grid_K[iKw];
        double Km = par->grid_K[iKm];

        // objective function
        int const dim = 2; // consumption is done conditional on labor
        double lb[dim],ub[dim],x[dim];
        
        auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT
        double minf=0.0;

        // bounds
        lb[0] = 0.0;
        ub[0] = 1.0;
        lb[1] = 0.0;
        ub[1] = 1.0;
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        solver_couple_struct* solver_data = new solver_couple_struct;
        solver_data->iP = iP;
        solver_data->love = love;
        solver_data->A = A;
        solver_data->Kw = Kw;
        solver_data->Km = Km;
        solver_data->t = t;
        solver_data->EVw_next = EVw_next;
        solver_data->EVm_next = EVm_next;
        solver_data->sol = sol;
        solver_data->par = par;
        solver_data->lower = lb;
        solver_data->upper = ub;

        solver_data->cons = 0.5; // initial guess on consumption (used in inner optimization)

        double Vw,Vm; // store individual values herein
        solver_data->Vw = &Vw; 
        solver_data->Vm = &Vm; 
        nlopt_set_min_objective(opt, objfunc_labor, solver_data);
        nlopt_set_ftol_rel(opt,1.0e-7);
        nlopt_set_xtol_rel(opt,1.0e-5);
        nlopt_set_maxeval(opt,250); // ensures finite termination

        // optimize
        x[0] = 0.5;
        x[1] = 0.5;
        int idx_last = -1;
        if(iKm>0){
            idx_last = index::couple(t,iP,iL,iA,iKw,iKm-1,par); 
        } else if(iKw>0) {
            idx_last = index::couple(t,iP,iL,iA,iKw-1,iKm,par); 
        }
        if(idx_last>-1){
            x[0] = sol->labor_w_remain_couple[idx_last];
            x[1] = sol->labor_m_remain_couple[idx_last];
        }
        nlopt_optimize(opt, x, &minf);
        nlopt_destroy(opt);

        // store results. 
        sol->labor_w_remain_couple[idx] = x[0];
        sol->labor_m_remain_couple[idx] = x[1];
        sol->cons_w_remain_couple[idx] = solver_data->cons;
        sol->cons_m_remain_couple[idx] = solver_data->cons;

        sol->Vw_remain_couple[idx] = Vw;
        sol->Vm_remain_couple[idx] = Vm;
        
    }


    void solve_couple(int t,sol_struct *sol,par_struct *par){
        
        #pragma omp parallel num_threads(par->threads)
        {   
            // allocate memory
            int const num = 6;
            double** list_start_as_couple = new double*[num]; 
            double** list_remain_couple = new double*[num];
            double* list_trans_to_single = new double[num];             

            double* Sw = new double[par->num_power];
            double* Sm = new double[par->num_power];

            index::index_couple_struct* idx_couple = new index::index_couple_struct;

            // loop through states (par.T,par.num_power,par.num_love,par.num_A,par.num_K,par.num_K)
            #pragma omp for
            for (int iP=0; iP<par->num_power; iP++){
                for (int iL=0; iL<par->num_love; iL++){
                    // Get next period continuation values
                    double *EVw_next = nullptr;  
                    double *EVm_next = nullptr;
                    if (t<(par->T-1)){
                        int idx_next = index::precomp(iP,iL,0,0,0,par); // interpolate next-period expected value (pre-computed) wrt. Anext,Kwbar,Kmbar (love is on grid)
                        EVw_next = &sol->EVw_pd[idx_next];  
                        EVm_next = &sol->EVm_pd[idx_next];
                    }
                    
                    for (int iA=0; iA<par->num_A; iA++){
                        for (int iKw=0; iKw<par->num_K; iKw++){
                            for (int iKm=0; iKm<par->num_K; iKm++){

                                solve_remain(t,iP,iL,iA,iKw,iKm,EVw_next,EVm_next,sol,par); 

                            } // human capital, man
                        } // human capital, woman
                    } // wealth
                } // love
            } // power

            // Solve for values of starting as couple (check participation constraints)
            #pragma omp for
            for (int iL=0; iL<par->num_love; iL++){
                for (int iA=0; iA<par->num_A; iA++){
                    for (int iKw=0; iKw<par->num_K; iKw++){
                        for (int iKm=0; iKm<par->num_K; iKm++){
                            // indices
                            int idx_single_w = index::single(t,iA,iKw,par);
                            int idx_single_m = index::single(t,iA,iKm,par);
                            
                            idx_couple->t = t;
                            idx_couple->iL = iL;
                            idx_couple->iA = iA;
                            idx_couple->iKw = iKw;
                            idx_couple->iKm = iKm;
                            idx_couple->par = par;

                            // Calculate marital surplus across power
                            for (int iP=0; iP<par->num_power; iP++){
                                int idx_tmp = index::couple(t,iP,iL,iA,iKw,iKm,par);
                                Sw[iP] = calc_marital_surplus(sol->Vw_remain_couple[idx_tmp],sol->Vw_single[idx_single_w],par);
                                Sm[iP] = calc_marital_surplus(sol->Vm_remain_couple[idx_tmp],sol->Vm_single[idx_single_m],par);
                            }

                            // setup relevant lists to pass to the bargaining algorithm
                            int i = 0;
                            list_start_as_couple[i] = sol->Vw_couple; i++;
                            list_start_as_couple[i] = sol->Vm_couple; i++;
                            list_start_as_couple[i] = sol->labor_w_couple; i++;
                            list_start_as_couple[i] = sol->labor_m_couple; i++;
                            list_start_as_couple[i] = sol->cons_w_couple; i++; 
                            list_start_as_couple[i] = sol->cons_m_couple; i++; 
                            i = 0;
                            list_remain_couple[i] = sol->Vw_remain_couple; i++;
                            list_remain_couple[i] = sol->Vm_remain_couple; i++;
                            list_remain_couple[i] = sol->labor_w_remain_couple; i++;
                            list_remain_couple[i] = sol->labor_m_remain_couple; i++;
                            list_remain_couple[i] = sol->cons_w_remain_couple; i++; 
                            list_remain_couple[i] = sol->cons_m_remain_couple; i++; 
                            i = 0;
                            list_trans_to_single[i] = sol->Vw_single[idx_single_w]; i++;
                            list_trans_to_single[i] = sol->Vm_single[idx_single_m]; i++;
                            list_trans_to_single[i] = sol->labor_w_single[idx_single_w]; i++;
                            list_trans_to_single[i] = sol->labor_m_single[idx_single_m]; i++;
                            list_trans_to_single[i] = sol->cons_w_single[idx_single_w]; i++; 
                            list_trans_to_single[i] = sol->cons_m_single[idx_single_m]; i++; 

                            // Update solutions in list_start_as_couple
                            if(par->bargaining==1){ // limited commitment

                                bargaining::check_participation_constraints(sol->power_idx, sol->power, Sw, Sm, idx_couple, list_start_as_couple, list_remain_couple, list_trans_to_single, num, par);
                            
                            } else if(par->bargaining==2){ // no commitment, Nash (equal weight, discrete)

                                bargaining::nash(sol->power_idx, sol->power, Sw, Sm, idx_couple, list_start_as_couple, list_remain_couple, list_trans_to_single, num, par);
                                // double obj_max = -1.0e10;
                                // int iP_max = -1;
                                // for (int iP=0; iP<par->num_power; iP++){

                                //     if((Sw[iP]>0.0) & (Sm[iP]>0.0)){
                                        
                                //         double obj_now = sqrt(Sw[iP]) * sqrt(Sm[iP]);
                                        
                                //         if(obj_now>obj_max){
                                //             obj_max = obj_now;
                                //             iP_max = iP;
                                //         }
                                //     }
                                // }

                                // if(iP_max>-1){
                                //     int idx_max = idx_couple->idx(iP_max);
                                //     int idx = idx_couple->idx(iP);
                                //     for (int i=0;i<num;i++){
                                //         list_start_as_couple[i][idx] = list_remain_couple[i][idx_max];
                                //     }

                                //     sol->power_idx[idx] = iP_max;
                                //     sol->power[idx] = par->grid_power[iP_max];

                                // } else {
                                //     // divorce
                                //     for (int i=0;i<num;i++){
                                //         list_start_as_couple[i][idx_tmp] = list_trans_to_single[i];
                                //     }

                                //     sol->power_idx[idx] = -1;
                                //     sol->power[idx] = -1.0;
                                // }


                            
                            } else { // no bargaining - full commitment

                                for (int iP=0; iP<par->num_power; iP++){
                                    int idx_tmp = idx_couple->idx(iP);
                                    for(int i=0;i<num;i++){
                                        list_start_as_couple[i][idx_tmp] = list_remain_couple[i][idx_tmp];
                                    }
                                }

                            } // bargaining_model check

                        } // human capital, man
                    } // human capital, woman
                } // wealth
            } // love

            
            // delete pointers
            delete[] list_start_as_couple;
            delete[] list_remain_couple;
            delete list_trans_to_single;

            delete Sw;
            delete Sm;

        } // pragma
    }
    
}
