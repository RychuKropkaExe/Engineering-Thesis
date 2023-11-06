#include <iostream>
#include <stdio.h>

int main(){
    std::cout.sync_with_stdio(false);
    printf("SIEMA ENIU 2\n");
    std::cout << "SIEMA ENIU\n";
    std::cout << "test" << std::flush;
}