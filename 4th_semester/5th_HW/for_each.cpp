#include <algorithm>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

template < typename Iterator, typename Function >
void my_for_each(Iterator first, Iterator last, Function function){
    const std::size_t length = std::distance(first, last);

    const std::size_t max_size = 32;

    if (length <= max_size) {
        std::for_each(first, last, function);
    }
    else {
        Iterator middle = first;
        std::advance(middle, length / 2);

        std::future < void > first_half_result = std::async(my_for_each<Iterator, Function >, first, middle, function);

        my_for_each(middle, last, function);

        first_half_result.get();
    }
}

int main(){
    std::vector < int > test(1e3);
    
    std::iota(test.begin(), test.end(), 0);

    my_for_each(test.begin(), test.end(), [](auto & x){x = x*x;});

    for(auto i : test) std::cout << i << " ";
    return 0;
}
