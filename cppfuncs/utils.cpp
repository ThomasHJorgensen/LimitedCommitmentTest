// functions related to utility and environment.
#ifndef MAIN
#define UTILS
#include "myheader.cpp"
#endif

namespace utils {
    double util(double cons,double labor ,int gender,par_struct *par){
        double rho = par->rho_w;
        double phi = par->phi_w;
        double alpha = par->alpha_w;
        if (gender == man) { 
            rho = par->rho_m;
            phi = par->phi_m;
            alpha = par->alpha_m;
        }
        
        double util_cons = pow(cons,1.0-rho)/(1.0-rho);
        double util_labor = - alpha*pow(labor,1.0+phi)/(1.0+phi);
        return  util_cons + util_labor;
    }


    double wage_func(double K, int gender, par_struct* par){
        double wage_const = par->wage_const_w;
        double wage_K = par->wage_K_w;
        if (gender==man){
            wage_const = par->wage_const_m;
            wage_K = par->wage_K_m; 
        }

        return exp(wage_const + wage_K*K);
    }

    double K_bar(double K, double labor, par_struct* par){
        return (1.0-par->K_depre)*K + labor;
    }

}