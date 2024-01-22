// functions related to utility and environment.
#ifndef MAIN
#define UTILS
#include "myheader.cpp"
#endif

namespace utils {
    // User-specified functions
    double equiv_scale(double cons, double d) {
        return cons / (1.0 + 0.7 * (1.0 - d));
    }

    double util(double cons,double labor ,int gender, double d, par_struct *par){
        double gamma1 = par->gamma1_w;
        double gamma2 = par->gamma2_w;
        if (gender == man) {
            gamma1 = par->gamma1_w;
            gamma2 = par->gamma2_w;
        }

        double C_public = equiv_scale(cons, d);

        return  (pow(C_public * exp(gamma2 * (1 - labor)), 1 - gamma1)) / (1 - gamma1) ;
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

    double tax_func(double income, par_struct *par) {
        double kappa1 = par->kappa1;
        double kappa2 = par->kappa2;

        return (1.0 - kappa1) * pow(income, 1.0 - kappa2);
    }

    double K_bar(double K, double labor, par_struct* par){
        return (1.0-par->K_depre)*K + par->lambdaa2*labor;
        //return (1.0-par->K_depre)*K + par->lambdaa2*labor;
    }

}