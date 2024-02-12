#ifndef MAIN
#define SINGLE
#include "myheader.h"
#endif
// TODO: room for speed improvements but the couples problem is the bottleneck
namespace single {
    typedef struct {
        
        double A;
        double K;   
        double Z;
        double *grid_weight_Z;         
        double *V_next;      
        int gender;  
        double labor; 
        double cons; 
        int t; 
        double *lower;
        double *upper;        
        par_struct *par;    

        

    } solver_single_struct;

    
    EXPORT double resources_single(double labor,double A,double K,int gender,par_struct* par) {
        double income = labor * utils::wage_func(K,gender,par);
        double after_tax_income = utils::tax_func(income,par);
        return par->R*A + after_tax_income;
    }

    double value_of_choice(double cons, double labor, double Z, double A, double K, int gender,int t, double* V_next, double* grid_weight_Z, par_struct* par){
        double d = 1.0; //Divorce
        // flow-utility
        double Util = utils::util(cons,labor,gender,d,par)-d*Z;
        // continuation value
        double EVnext = 0.0;
        if(t<par->T-1){
            double *grid_A = par->grid_Aw; 
            if (gender==man){
                grid_A = par->grid_Am;
            }

            // Expected continuation value
            double A_next = resources_single(labor,A,K,gender,par) - cons;
            double Kbar = utils::K_bar(K,labor,t,par);
            // expected Z
            for(int iK_next=0;iK_next<par->num_shock_K;iK_next++){
                for(int iZ_next=0;iZ_next<par->num_Z;iZ_next++){
                    int idx_next = index::index3(iZ_next,0,0, par->num_Z,par->num_A,par->num_K); //virker denne?
                    double K_next = Kbar*par->grid_shock_K[iK_next];
                    double Znext = iZ_next;
                    
                    double interp_next = tools::interp_2d(grid_A,par->grid_K,par->num_A,par->num_K,V_next,A_next,K_next);
                    EVnext += par->grid_weight_K[iK_next] * grid_weight_Z[iZ_next] *interp_next;
                }
            }
        }

        // return discounted sum
        return Util + par->beta*EVnext;
    }

    double objfunc_single_cons(unsigned n, const double *x, double *grad, void *solver_data_in){

        // unpack
        solver_single_struct *solver_data = (solver_single_struct *) solver_data_in;  
        
        double cons = x[0];
        int gender = solver_data->gender;
        double A = solver_data->A;
        double K = solver_data->K;
        double Z = solver_data->Z;
        double* grid_weight_Z = solver_data->grid_weight_Z ;
        double labor = solver_data->labor;
        double* V_next = solver_data->V_next;
        int t = solver_data->t;
        par_struct *par = solver_data->par;

        // penalty and clip
        double penalty = 0.0;
        double saving = resources_single(labor,A,K,gender,par) - cons;
        if(saving<0.0){ // budget constraint: no borrowing
            penalty += 1000.0*saving*saving;
            cons = resources_single(labor,A,K,gender,par);  
        }

        // return negative value of choice
        return - value_of_choice(cons,labor,Z,A,K,gender,t,V_next,grid_weight_Z,par) + penalty;

    }
    double objfunc_single_labor(unsigned n, const double *x, double *grad, void *solver_data_in){

        // unpack
        solver_single_struct *solver_data = (solver_single_struct *) solver_data_in;  
        
        double labor = x[0];
        int gender = solver_data->gender;
        double A = solver_data->A;
        double K = solver_data->K;
        par_struct *par = solver_data->par;
        solver_data->labor = labor;

        // penalty and clip
        double penalty = 0.0;
        if (labor < solver_data->lower[0]) {
            penalty += 1000.0*(solver_data->lower[0]-labor)*(solver_data->lower[0]-labor);
            labor = solver_data->lower[0];
        } else if (labor > solver_data->upper[0]) {
            penalty += 1000.0*(solver_data->upper[0]-labor)*(solver_data->upper[0]-labor);
            labor = solver_data->upper[0];
        }

        // solve for optimal consumption at this level of labor supply
        int const dim = 1;
        double lb[dim],ub[dim],y[dim];
        
        auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); // NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT
        double minf=0.0;

        // bounds
        lb[0] = 1.0e-6;
        ub[0] = resources_single(labor,A,K,gender,par)-1.0e-6;
        nlopt_set_lower_bounds(opt, lb);
        nlopt_set_upper_bounds(opt, ub);

        nlopt_set_ftol_rel(opt,1.0e-5);
        nlopt_set_xtol_rel(opt,1.0e-5);

        
        nlopt_set_min_objective(opt, objfunc_single_cons, solver_data); 

        // optimize
        y[0] = MIN(ub[0],0.5); 
        nlopt_optimize(opt, y, &minf); 
        solver_data->cons = y[0];

        // destroy optimizer
        nlopt_destroy(opt);

        // return objective function
        return minf + penalty;

    }

    void solve_single(int t, sol_struct *sol,par_struct *par){

        #pragma omp parallel num_threads(par->threads)
        {

            // allocate objects for solver
            solver_single_struct* solver_data = new solver_single_struct;
            
            int const dim = 1; // only labor supply in outer. Consumption in inner
            double lb[dim],ub[dim],x[dim];
            
            auto opt = nlopt_create(NLOPT_LN_BOBYQA, dim); //NLOPT_LN_BOBYQA NLOPT_LD_MMA NLOPT_LD_LBFGS NLOPT_GN_ORIG_DIRECT
            nlopt_set_ftol_rel(opt,1.0e-8);
            nlopt_set_xtol_rel(opt,1.0e-6);
            double minf=0.0;

            int idx_next = index::index4(t+1,0,0,0,par->T, par->num_Z,par->num_A,par->num_K);
            int idx_last = 0;

            #pragma omp for
            for (int iA=0; iA<par->num_A;iA++){
                for (int iK=0; iK<par->num_K;iK++){
                    for (int iZ=0; iZ<par->num_Z;iZ++){
                        int idx = index::single(t,iZ,iA,iK,par);//index::index3(t,iA,iK,par->T,par->num_A,par->num_K);
                        int idx_Z = index::index2(iZ,0,par->num_Z,par->num_Z);

                        // states
                        double Aw = par->grid_Aw[iA];
                        double Am = par->grid_Am[iA];
                        
                        double Zw = par->grid_Z[iZ];
                        double Zm = par->grid_Z[iZ];

                        double Kw = par->grid_K[iK];
                        double Km = par->grid_K[iK];

                        // WOMEN
                        // bounds
                        lb[0] = 0.0;
                        ub[0] = 1.0;    
                        nlopt_set_lower_bounds(opt, lb);
                        nlopt_set_upper_bounds(opt, ub);

                        // settings
                        solver_data->A = Aw;
                        solver_data->K = Kw;
                        solver_data->Z = Zw;
                        solver_data->grid_weight_Z = &par->grid_weight_Z[idx_Z];
                        solver_data->gender = woman;
                        solver_data->lower = lb;
                        solver_data->upper = ub;
                        solver_data->par = par;
                        solver_data->V_next = &sol->Vw_single[idx_next];
                        solver_data->t = t;
                        nlopt_set_min_objective(opt, objfunc_single_labor, solver_data); 

                        
                        // optimize
                        x[0] = 0.5;  
                        idx_last = -1;
                        if(iK>0){
                            idx_last = index::single(t,iZ,iA,iK-1,par);
                        } 
                        if(idx_last>-1){
                            x[0] = sol->labor_w_single[idx_last];
                        }
                        nlopt_optimize(opt, x, &minf); 

                        // store results
                        sol->labor_w_single[idx] = x[0];
                        sol->cons_w_single[idx] = solver_data->cons;
                        sol->Vw_single[idx] = -minf;

                        sol->labor_w_trans_single[idx] = sol->labor_w_single[idx];
                        sol->cons_w_trans_single[idx] = sol->cons_w_single[idx];
                        sol->Vw_trans_single[idx] = sol->Vw_single[idx] - par->div_cost;

                        // MEN
                        // bounds
                        lb[0] = 0.0;
                        ub[0] = 1.0;
                        nlopt_set_lower_bounds(opt, lb);
                        nlopt_set_upper_bounds(opt, ub);

                        // settings
                        solver_data->A = Am;
                        solver_data->K = Km;
                        solver_data->Z = Zm;
                        solver_data->grid_weight_Z = &par->grid_weight_Z[idx_Z];
                        solver_data->gender = man;
                        solver_data->lower = lb;
                        solver_data->upper = ub;
                        solver_data->par = par;
                        solver_data->V_next = &sol->Vm_single[idx_next];
                        solver_data->t = t;
                        nlopt_set_min_objective(opt, objfunc_single_labor, solver_data);  

                        // optimize
                        x[0] = 0.5;
                        if(idx_last>-1){
                            x[0] = sol->labor_m_single[idx_last];
                        }
                        nlopt_optimize(opt, x, &minf); 

                        // store results
                        sol->labor_m_single[idx] = x[0];
                        sol->cons_m_single[idx] = solver_data->cons;
                        sol->Vm_single[idx] = -minf;

                        sol->labor_m_trans_single[idx] = sol->labor_m_single[idx];
                        sol->cons_m_trans_single[idx] = sol->cons_m_single[idx];
                        sol->Vm_trans_single[idx] = sol->Vm_single[idx] - par->div_cost;
                    }
                }
            }

            // destroy optimizer
            nlopt_destroy(opt);

        } // pragma
        
    } // solve 

} // namespace single
