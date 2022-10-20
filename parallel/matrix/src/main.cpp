#include <iostream>
#include <vector>
#include <fstream>
#include "math.h"
#include "matrix.hpp"

#include <chrono>
#include <cassert>

#define assertm(exp, msg)

using namespace std;

int main(int argc, char *argv[]){
    
    assertm(argc==3, "Wrong arguments, use : ./bin/start N operation(s or m)");
    
    const unsigned int p = stoi(argv[1]);
    const char operation = argv[2][0];
    
    unsigned int count = 10;
    
    for(unsigned int n = 10; n <= p; n+=count)
        {
        
        matrix Matrix1(n,n);
        matrix Matrix2(n,n);
        
        for (unsigned int i = 0; i < n; i++){
            for (unsigned int j = 0; j < n; j++){
                Matrix1.SetIJComp(1,i,j);
                Matrix2.SetIJComp(2,i,j);
            }
        }for (unsigned int i = 0; i < n; i++){
            for (unsigned int j = 0; j < n; j++){
                Matrix1.SetIJComp(1,i,j);
                Matrix2.SetIJComp(2,i,j);
            }
        }
        
        if (operation == 's')
        {
            matrix MatrixSum(n,n);
            
            auto start = std::chrono::steady_clock::now();

            MatrixSum = Matrix1 + Matrix2;

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            std::cout  << n*n*8/1024 << " " << 1000*elapsed_seconds.count() << endl;
            
        }

        if (operation == 'm')
        {
            matrix MatrixSum(n,n);
            
            auto start = std::chrono::steady_clock::now();

            MatrixSum = Matrix1 * Matrix2;

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            std::cout << n << " " << n*n*8/1024 << " " << 1000*elapsed_seconds.count() << endl;
            
        }
    }
	
	

	return 0;
}
