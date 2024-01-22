#include <windows.h>
#define EXPORT extern "C" __declspec(dllexport)
EXPORT void setup_omp(){
SetEnvironmentVariable("OMP_WAIT_POLICY", "passive");
}
