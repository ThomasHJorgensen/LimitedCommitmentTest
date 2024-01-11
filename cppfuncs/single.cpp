#ifndef MAIN
#define SINGLE
#include "myheader.cpp"
#endif

namespace single {
    typedef struct {
        
        double A;
        double K;             
        double *V_next;      
        int gender;  
        double *lower;
        double *upper;        
        par_struct *par;      

    } solver_single_struct;

    
    EXPORT double resources_single(double labor,double A,double K,int gender,par_struct* par) {
        double income = labor * utils::wage_func(K,gender,par);
        return par->R*A + income;
    }

    double value_of_choice(double cons, double labor,double A, double K, int gender, double* V_next, par_struct* par){

        // flow-utility
        double Util = utils::util(cons,labor,gender,par);
        
        // continuation value
        double *grid_A = par->grid_Aw; 
        if (gender==man){
            grid_A = par->grid_Am;
        }

        // Expected continuation value
        double A_next = resources_single(labor,A,K,gender,par) - cons;
        double Kbar = utils::K_bar(K,labor,par);

        double EVnext = 0.0;
        for(int iK_next=0;iK_next<par->num_shock_K;iK_next++){
            double K_next = Kbar*par->grid_shock_K[iK_next];
            
            double interp_next = tools::interp_2d(grid_A,par->grid_K,par->num_A,par->num_K,V_next,A_next,K_next);
            EVnext += par->grid_weight_K[iK_next] * interp_next;
        }

        // return discounted sum
        return Util + par->beta*EVnext;
    }

    double objfunc_single_last(unsigned n, const double *x, double *grad, void *solver_data_in){

        // unpack
        solver_single_struct *solver_data = (solver_single_struct *) solver_data_in;  
        
        double labor = x[0];
        int gender = solver_data->gender;
        double A = solver_data->A;
        double K = solver_data->K;
        par_struct *par = solver_data->par;

        // penalty and clip
        double penalty = 0.0;
        if (labor < solver_data->lower[0]) {
            penalty += 1000.0*(solver_data->lower[0]-labor)*(solver_data->lower[0]-labor);
            labor = solver_data->lower[0];
        } else if (labor > solver_data->upper[0]) {
            penalty += 1000.0*(solver_data->upper[0]-labor)*(solver_data->upper[0]-labor);
            labor = solver_data->upper[0];
        }
            
        // consume all available resources_single
        double cons = resources_single(labor,A,K,gender,par);
        double util = utils::util(cons,labor,gender,par);

        return - util + penalty;

    }

    double objfunc_single(unsigned n, const double *x, double *grad, void *solver_data_in){

        // unpack
        solver_single_struct *solver_data = (solver_single_struct *) solver_data_in;  
        
        double labor = x[0];
        double cons = x[1];
        int gender = solver_data->gender;
        double A = solver_data->A;
        double K = solver_data->K;
        par_struct *par = solver_data->par;

        // penalty and clip
        double penalty = 0.0;
        if (labor < solver_data->lower[0]) {
            penalty += 1000.0*(solver_data->lower[0]-labor)*(solver_data->lower[0]-labor);
            labor = solver_data->lower[0];
        } else if (labor > solver_data->upper[0]) {
            penalty += 1000.0*(solver_data->upper[0]-labor)*(solver_data->upper[0]-labor);
            labor = solver_data->upper[0];
        }

        double saving = resources_single(labor,A,K,gender,par) - cons;
        if(saving<0.0){ // budget constraint: no borrowing
            penalty += 1000.0*saving*saving;
            cons -= saving; 
        }
        if (cons < solver_data->lower[1]) {
            penalty += 1000.0*(solver_data->lower[1]-cons)*(solver_data->lower[1]-cons);
            cons = solver_data->lower[1];
        } else if (cons > solver_data->upper[1]) {
            penalty += 1000.0*(solver_data->upper[1]-cons)*(solver_data->upper[1]-cons);
            cons = solver_data->upper[1];
        }

        // return negative value of choice
        return - value_of_choice(cons,labor,A,K,gender,solver_data->V_next,par) + penalty;

    }

    void solve_single_last(sol_struct *sol,par_struct *par){

        #pragma omp parallel num_threads(par->threads)
        {
            int t = par->T-1;

            // allocate objects for solver
            solver_single_struct* solver_data = new solver_single_struct;
            
            int const dim = 1;
            double lb[dim],ub[dim],x[dim];
            
            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT
            double minf=0.0;

            // bounds
            lb[0] = 0.0;
            ub[0] = 1.0;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            #pragma omp for
            for (int iA=0; iA<par->num_A;iA++){
                for (int iK=0; iK<par->num_K;iK++){
                    int idx = index::single(t,iA,iK,par);//index::index3(t,iA,iK,par->T,par->num_A,par->num_K);

                    // states
                    double Aw = par->grid_Aw[iA];
                    double Am = par->grid_Am[iA];

                    double Kw = par->grid_K[iK];
                    double Km = par->grid_K[iK];

                    // WOMEN
                    // settings
                    solver_data->A = Aw;
                    solver_data->K = Kw;
                    solver_data->gender = woman;
                    solver_data->lower = lb;
                    solver_data->upper = ub;
                    solver_data->par = par;
                    nlopt_set_min_objective(opt, objfunc_single_last, solver_data); 

                    // optimize
                    x[0] = 0.5; 
                    if(iK>0){
                        x[0] = sol->labor_w_single[index::index3(t,iA,iK-1,par->T,par->num_A,par->num_K)];
                    }
                    nlopt_optimize(opt, x, &minf); 

                    // store results
                    sol->labor_w_single[idx] = x[0];
                    sol->cons_w_single[idx] = resources_single(sol->labor_w_single[idx],Aw,Kw,woman,par);
                    sol->Vw_single[idx] = -minf;

                    // MEN
                    // settings
                    solver_data->A = Am;
                    solver_data->K = Km;
                    solver_data->gender = man;                    
                    solver_data->lower = lb;
                    solver_data->upper = ub;
                    solver_data->par = par;
                    nlopt_set_min_objective(opt, objfunc_single_last, solver_data); 

                    // optimize
                    x[0] = 0.5; 
                    if(iK>0){
                        x[0] = sol->labor_m_single[index::index3(t,iA,iK-1,par->T,par->num_A,par->num_K)];
                    }
                    nlopt_optimize(opt, x, &minf); 

                    // store results
                    sol->labor_m_single[idx] = x[0];
                    sol->cons_m_single[idx] = single::resources_single(sol->labor_m_single[idx],Am,Km,man,par);
                    sol->Vm_single[idx] = -minf;

                }
            }
            // destroy optimizer
            nlopt_destroy(opt);

        } // pragma
        
    } // solve last


    void solve_single(int t, sol_struct *sol,par_struct *par){

        #pragma omp parallel num_threads(par->threads)
        {

            // allocate objects for solver
            solver_single_struct* solver_data = new solver_single_struct;
            
            int const dim = 2;
            double lb[dim],ub[dim],x[dim];
            
            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT
            double minf=0.0;

            int idx_next = index::index3(t+1,0,0,par->T,par->num_A,par->num_K);
            int idx_last = 0;

            #pragma omp for
            for (int iA=0; iA<par->num_A;iA++){
                for (int iK=0; iK<par->num_K;iK++){
                    int idx = index::single(t,iA,iK,par);//index::index3(t,iA,iK,par->T,par->num_A,par->num_K);

                    // states
                    double Aw = par->grid_Aw[iA];
                    double Am = par->grid_Am[iA];

                    double Kw = par->grid_K[iK];
                    double Km = par->grid_K[iK];

                    // WOMEN
                    // bounds
                    lb[0] = 0.0;
                    ub[0] = 1.0;
                    lb[1] = 1.0e-6; // consumption
                    ub[1] = resources_single(ub[0],Aw,Kw,woman,par); // resources_single if working full time 
                    nlopt_set_lower_bounds(opt, lb);
                    nlopt_set_upper_bounds(opt, ub);

                    // settings
                    solver_data->A = Aw;
                    solver_data->K = Kw;
                    solver_data->gender = woman;
                    solver_data->lower = lb;
                    solver_data->upper = ub;
                    solver_data->par = par;
                    solver_data->V_next = &sol->Vw_single[idx_next];
                    nlopt_set_min_objective(opt, objfunc_single, solver_data); 

                    
                    // optimize
                    x[0] = 0.5; 
                    x[1] = 0.5*ub[1];
                    if(iK>0){
                        idx_last = index::index3(t,iA,iK-1,par->T,par->num_A,par->num_K);
                        x[0] = sol->labor_w_single[idx_last];
                        x[1] = sol->cons_w_single[idx_last];
                    }
                    nlopt_optimize(opt, x, &minf); 

                    // store results
                    sol->labor_w_single[idx] = x[0];
                    sol->cons_w_single[idx] = x[1];
                    sol->Vw_single[idx] = -minf;

                    sol->labor_w_trans_single[idx] = sol->labor_w_single[idx];
                    sol->cons_w_trans_single[idx] = sol->cons_w_single[idx];
                    sol->Vw_trans_single[idx] = sol->Vw_single[idx] - par->div_cost;

                    // MEN
                    // bounds
                    lb[0] = 0.0;
                    ub[0] = 1.0;
                    lb[1] = 1.0e-6; // consumption
                    ub[1] = resources_single(ub[0],Am,Km,man,par); // resources_single if working full time 
                    nlopt_set_lower_bounds(opt, lb);
                    nlopt_set_upper_bounds(opt, ub);

                    // settings
                    solver_data->A = Am;
                    solver_data->K = Km;
                    solver_data->gender = man;
                    solver_data->lower = lb;
                    solver_data->upper = ub;
                    solver_data->par = par;
                    solver_data->V_next = &sol->Vm_single[idx_next];
                    nlopt_set_min_objective(opt, objfunc_single, solver_data); 

                    // optimize
                    x[0] = 0.5;
                    x[1] = 0.5*ub[1]; 
                    if(iK>0){
                        x[0] = sol->labor_m_single[idx_last];
                        x[1] = sol->cons_m_single[idx_last];
                    }
                    nlopt_optimize(opt, x, &minf); 

                    // store results
                    sol->labor_m_single[idx] = x[0];
                    sol->cons_m_single[idx] = x[1];
                    sol->Vm_single[idx] = -minf;

                    sol->labor_m_trans_single[idx] = sol->labor_m_single[idx];
                    sol->cons_m_trans_single[idx] = sol->cons_m_single[idx];
                    sol->Vm_trans_single[idx] = sol->Vw_single[idx] - par->div_cost;

                }
            }

            // destroy optimizer
            nlopt_destroy(opt);

        } // pragma
        
    } // solve 

} // namespace single
