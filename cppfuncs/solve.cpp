#define MAIN
#include "myheader.h"

// include these again here to ensure that they are automatically compiled by consav
#ifndef MAIN
#include "single.cpp"
#endif


/////////////
// 5. MAIN //
/////////////

EXPORT void solve(sol_struct *sol, par_struct *par){
    
    // loop backwards
    for (int t = par->T-1; t >= 0; t--){

        single::solve_single(t,sol,par); 
        
        if(t<(par->T-1)){
            couple::precompute_EV(t+1,sol,par);
        }
        couple::solve_couple(t,sol,par);

    }
}

EXPORT void simulate(sim_struct *sim, sol_struct *sol, par_struct *par){
    
    sim::model(sim,sol,par);

}

EXPORT double calc_bargaining_weight(int t, double love, double Aw,double Am, double Kw, double Km, int iZw, int iZm, sol_struct* sol, par_struct* par){
    return bargaining::calc_bargaining_weight(t,love,Aw,Am,Kw,Km,iZw,iZm,sol,par);
}


