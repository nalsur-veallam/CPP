#include <iostream>
#include <chrono>

#define N 100000

int main(int argc, char *argv[]){
    
    int a = 0;
    double time_m = 0;
    double time_a = 0;
    auto start = std::chrono::steady_clock::now();
    
    for(int i = 1; i <= N; i++){
        a += 1;
    }
    
    auto end = std::chrono::steady_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end-start;
    time_a += double(elapsed_seconds.count());
    
    int m = 0;
    auto start = std::chrono::steady_clock::now();
    
    for(int i = 1; i <= N; i++){
        m *= 1;
    }
    
    auto end = std::chrono::steady_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end-start;
    time_m += double(elapsed_seconds.count());
    
    std::cout << "Addition time: " << 1000/N*time_a << std::endl;
    std::cout << "Multiply time: " << 1000/N*time_m << std::endl;
    
    return 0;
}
