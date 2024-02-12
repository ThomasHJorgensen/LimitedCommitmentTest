// functions for calculating linear indices.
#ifndef MAIN
#define INDEX
#include "myheader.h"
#endif

namespace index {
    int index2(int i1,int i2,int N1,int N2){
        return i2 + i1*N2;
    }
    int index3(int i1,int i2,int i3,int N1,int N2, int N3){
        return i3 + i2*N3 + i1*N2*N3;
    }
    int index4(int i1,int i2,int i3,int i4,int N1,int N2, int N3, int N4){
        return i4 + (i3 + (i2 + i1*N2)*N3)*N4;
    }

    int index5(int i_x1, int i_x2, int i_x3, int i_x4, int i_x5, int Nx1,int Nx2, int Nx3, int Nx4, int Nx5){

        int i_grid = (((i_x1*Nx2
                        + i_x2)*Nx3
                        + i_x3)*Nx4
                        + i_x4)*Nx5
                        + i_x5;
        return i_grid;
    }

    int index6(int i_x1, int i_x2, int i_x3, int i_x4, int i_x5, int i_x6,int Nx1, int Nx2, int Nx3, int Nx4, int Nx5, int Nx6){

        int i_grid = ((((i_x1*Nx2
                        + i_x2)*Nx3
                        + i_x3)*Nx4
                        + i_x4)*Nx5
                        + i_x5)*Nx6
                        + i_x6;
        
        return i_grid;
    }

    int index7(int i_x1, int i_x2, int i_x3, int i_x4, int i_x5, int i_x6, int i_x7,int Nx1, int Nx2, int Nx3, int Nx4, int Nx5, int Nx6, int Nx7){

        int i_grid = (((((i_x1*Nx2
                        + i_x2)*Nx3
                        + i_x3)*Nx4
                        + i_x4)*Nx5
                        + i_x5)*Nx6
                        + i_x6)*Nx7
                        + i_x7;
        return i_grid;
    }

    int index8(int i_x1, int i_x2, int i_x3, int i_x4, int i_x5, int i_x6, int i_x7, int i_x8, int Nx1, int Nx2, int Nx3, int Nx4, int Nx5, int Nx6, int Nx7, int Nx8){

        int i_grid = ((((((i_x1*Nx2
                        + i_x2)*Nx3
                        + i_x3)*Nx4
                        + i_x4)*Nx5
                        + i_x5)*Nx6
                        + i_x6)*Nx7
                        + i_x7)*Nx8
                        + i_x8;
        return i_grid;
    }

    int single(int t, int iZ, int iA, int iK,par_struct* par){
        return index4(t,iZ, iA,iK,par->T,par->num_Z,par->num_A,par->num_K);
    }

    int couple(int t, int iZw, int iZm,int iP,int iL,int iA,int iKw,int iKm,par_struct* par){
        return index8(t,iZw,iZm,iP,iL,iA,iKw,iKm,par->T, par->num_Z, par->num_Z,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K);
    }
    int precomp(int iZw, int iZm,int iP,int iL,int iA,int iKw,int iKm,par_struct* par){
        return index7(iZw, iZm,iP,iL,iA,iKw,iKm,par->num_Z, par->num_Z, par->num_power,par->num_love,par->num_A_pd,par->num_K_pd,par->num_K_pd);
    }

    typedef struct{
            int t;
            int iZw;
            int iZm;
            int iL;
            int iA;
            int iKw;
            int iKm;
            par_struct *par; 
            int idx(int iP){
                    return couple(t,iZw,iZm,iP,iL,iA,iKw,iKm,par); //index::index6(t,iP,iL,iA,iKw,iKm , par->T,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K); 
            }
        
    } index_couple_struct;

    
}