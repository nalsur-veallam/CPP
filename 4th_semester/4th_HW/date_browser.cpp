#include <iostream>
#include <regex>
#include <fstream>
#include <vector>


int main() {
    std::regex pattern(R"((0[1-9]|[12][0-9]|3[01])[-:/.](0[1-9]|1[012])[- :/\.]([012]\d{3}))"); //

    std::ifstream input("data.txt");

    std::string buffer;
    std::vector<std::string> dates;
    while (std::getline(input, buffer)) {
        std::copy(
                std::sregex_token_iterator(buffer.begin(), buffer.end(), pattern, {0 }),
                std::sregex_token_iterator(),
                std::back_inserter(dates));
    }

    std::cout << "\nDates: ";
    std::copy(std::begin(dates), std::end(dates),
              std::ostream_iterator < std::string >(std::cout, " "));
    std::cout << "\n";
    return 0;
}
