#include <time.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>
#include <deque>
#include <list>
#include <forward_list>

const int N = 1000000;

int main () {
    //Creating an data structures with random numbers
    std::array<int,N> arr;
    std::vector<int> vec;
    std::deque<int> deq;
    std::list<int> l;
    std::forward_list<int> f_l;
    for(int i = 0; i < N; i++){
        arr[i] = rand();
        vec.push_back(arr[i]);
        deq.push_back(arr[i]);
        l.push_back(arr[i]);
        f_l.push_front(arr[i]);
    }
    //Count the time for array
    clock_t t_arr = clock();
    std::sort(arr.begin(), arr.end());
    std::cout << (static_cast<double>(clock() - t_arr) / CLOCKS_PER_SEC) << "sec for array" << '\n';
    //Count the time for vector
    clock_t t_vec = clock();
    std::sort(vec.begin(), vec.end());
    std::cout << (static_cast<double>(clock() - t_vec) / CLOCKS_PER_SEC) << "sec for vector" << '\n';
    //Count the time for deque
    clock_t t_deq = clock();
    std::sort(deq.begin(), deq.end());
    std::cout << (static_cast<double>(clock() - t_deq) / CLOCKS_PER_SEC) << "sec for deque" << '\n';
    //Count the time for list
    clock_t t_l = clock();
    l.sort();
    std::cout << (static_cast<double>(clock() - t_l) / CLOCKS_PER_SEC) << "sec for list" << '\n';
    clock_t t_fl = clock();
    //Count the time for forward_list
    f_l.sort();
    std::cout << (static_cast<double>(clock() - t_fl) / CLOCKS_PER_SEC) << "sec for forward_list" << '\n';
    return 0;
}
