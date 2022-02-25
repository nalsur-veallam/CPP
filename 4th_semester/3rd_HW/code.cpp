#include <numeric>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include <cstdlib>
#include <functional>
#include <iterator>


bool simple (int &N){
    bool rez=true;
    for(int k=2;k<=N/2;k++)
        if (N%k==0) rez=false;
    return rez;
}



int main(){
    // #1
    std::vector <int> vec(10);
    std::iota(vec.begin(), vec.end(), 1);

    // #2
    typedef std::istream_iterator<int> in_it;
    typedef std::ostream_iterator<int> out_it;
    std::cout << "Please enter sequence of an integers (to quit press q + Enter):\n";
    std::copy(in_it(std::cin), in_it(), std::back_inserter(vec));
    //std::copy(vec.begin(), vec.end(), out_it(std::cout, " "));

    // #3
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(vec.begin(), vec.end(), g);

    // #4
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());

    // #5
    std::cout << "There is "<< std::count_if(vec.begin(), vec.end(), [](auto elem){return elem%2 != 0;}) << " odd numbers in the array.\n";

    // #6
    std::cout << "Max element is " << *std::max_element(vec.begin(), vec.end())<< "; Min element is " << *std::min_element(vec.begin(), vec.end()) << '\n';

    // #7
    for(auto it  = vec.begin(); it != vec.end(); it++){
        if(simple(*it)){
            std::cout << "Prime number " << *it << " is found.\n";
            break;
        }
    }

    // #8
    std::transform(vec.begin(), vec.end(), vec.begin(), [](auto elem){return elem*elem;});

    // #9
    std::vector<int> vec2(vec.size(), 0);
    std::transform(vec2.begin(), vec2.end(), vec2.begin(), [](auto){return rand()%100;});

    // #10
    std::cout << "The sum of the elements of the second array is " << std::accumulate(vec2.begin(), vec2.end(), 0) << "\n";

    // #11
    std::transform(vec2.begin(), vec2.begin()+=4, vec2.begin(), [](auto){return 1;});

    // #12
    std::vector<int> vec3;
    std::transform(vec.begin(), vec.end(), vec2.begin(), std::back_inserter(vec3), std::minus<int>());

    // #13
    std::transform(vec3.begin(), vec3.end(), vec3.begin(), [](auto elem){return (elem > 0) ? elem :0 ;});

    // #14
    vec3.erase(std::remove(vec3.begin(), vec3.end(), 0), vec3.end());

    // #15
    std::reverse(vec3.begin(), vec3.end());

    // #16
    std::nth_element(vec3.begin(), vec3.end()-=4, vec3.end());
    std::copy(vec3.end()-=3, vec3.end(), out_it(std::cout, " "));
    std::cout << " - The three largest elements.\n";

    // #17
    std::sort(vec.begin(), vec.end());
    std::sort(vec2.begin(), vec2.end());

    // #18
    std::vector<int> vec4;
    std::merge(vec.begin(), vec.end(), vec2.begin(), vec2.end(), std::back_inserter(vec4));

    // #19
    std::cout << "Equal range for 1 is ";
    std::vector <int> :: iterator it_1, it_2;
    std::copy(std::equal_range(vec4.begin(), vec4.end(), 1).first, std::equal_range(vec4.begin(), vec4.end(), 1).second, out_it(std::cout, " "));
    std::cout << '\n';

    // #20
    std::cout << "First array: ";
    std::copy(vec.begin(), vec.end(), out_it(std::cout, " "));
    std::cout << '\n' << "Second array: ";
    std::copy(vec2.begin(), vec2.end(), out_it(std::cout, " "));
    std::cout << '\n' << "Third array: ";
    std::copy(vec3.begin(), vec3.end(), out_it(std::cout, " "));
    std::cout << '\n' << "Fourth array: ";
    std::copy(vec4.begin(), vec4.end(), out_it(std::cout, " "));
    return 0;
}

