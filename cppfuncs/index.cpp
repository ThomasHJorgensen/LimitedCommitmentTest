// functions for calculating linear indices.
#ifndef MAIN
#define INDEX
#include "myheader.cpp"
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
    int single(int t, int iA, int iK,par_struct* par){
        return index3(t,iA,iK,par->T,par->num_A,par->num_K);
    }

    int couple(int t,int iP,int iL,int iA,int iKw,int iKm,par_struct* par){
        return index6(t,iP,iL,iA,iKw,iKm,par->T,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K);
    }

    typedef struct{
            int t;
            int iL;
            int iA;
            int iKw;
            int iKm;
            par_struct *par; 
            int idx(int iP){
                    return couple(t,iP,iL,iA,iKw,iKm,par); //index::index6(t,iP,iL,iA,iKw,iKm , par->T,par->num_power,par->num_love,par->num_A,par->num_K,par->num_K); 
            }
        
    } index_couple_struct;

    
}