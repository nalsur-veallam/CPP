#include <iostream>
#include <string>
#include <map>
#include <random>
#include <unordered_map>


class PhoneBook{
private:
    std::map<std::string, std::string> map;
    std::unordered_map<std::string,std::string> unmap;
public:
    PhoneBook() {};

    void add_subscriber(std::string surname, std::string number){
        if (unmap.find(surname) != unmap.end()){
            std::cout << "A user named " << surname << " already exists. Change the name.\n";
        }
        else{
            unmap.insert(std::make_pair(surname, number));
            map.insert(std::make_pair(surname, number));
        }
    }

    //Access for typography.
    //Printed people with surname starting with first_surname and
    //ending with last_surname.
    //If you want to start from the beginning, then put
    //first_surname equal to -1; Same with last_surname.
    void find_for_typ(std::string first_surname, std::string last_surname){
        std::map <std::string, std::string> :: iterator it, it_1, it_2;
        if(first_surname == "-1" && last_surname == "-1"){
            print_all();
        }
        else if (last_surname == "-1"){
            map.insert(std::make_pair(first_surname + "-1", "-1"));
            it_1 = map.find(first_surname + "-1");
            it_1++;
            map.erase(map.find(first_surname + "-1"));
            for (it=it_1; it!=map.end(); it++)
                std::cout << it->first << ' ' << it->second << '\n';
        }
        else if (first_surname == "-1"){
            map.insert(std::make_pair(last_surname + "-1", "-1"));
            it_2 = map.find(last_surname + "-1");
            it_2--;
            map.erase(map.find(last_surname + "-1"));
            for (it=map.begin(); it!=it_2; it++)
                std::cout << it->first << ' ' << it->second << '\n';
        }
        else {
            map.insert(std::make_pair(last_surname + "-1", "-1"));
            map.insert(std::make_pair(first_surname + "-1", "-1"));
            it_1 = map.find(first_surname + "-1");
            it_1++;
            it_2 = map.find(last_surname + "-1");
            it_2--;
            map.erase(map.find(last_surname + "-1"));
            map.erase(map.find(first_surname + "-1"));
            for (it=it_1; it!=it_2; it++)
                std::cout << it->first << ' ' << it->second << '\n';
        }
    }

    void print_all(){
        std::map <std::string, std::string> :: iterator it;
        for (it=map.begin(); it!=map.end(); it++)
            std::cout << it->first << ' ' << it->second << '\n';
    }

    //Random access to directory entries (for an advertising agency).
    void find_random(){
        int s = map.size();
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_int_distribution<> distribution(1, s - 1);
        s = distribution(generator);
        std::map <std::string, std::string> :: iterator it;
        it = map.begin();
        for(int i = 0; i < s; i++)
            it++;
        std::cout << it->first << ' ' << it->second << '\n';
    }

    //Quick search by person's last name (for regular users).
    void find_quick(std::string surname){
        std::unordered_map <std::string, std::string> :: iterator it;
        it = unmap.find(surname);
        if(it != unmap.end())
            std::cout << it->first << ' ' << it->second << '\n';
        else
            std::cout << "Such a name does not exist.\n";
    }
    ~PhoneBook() {};
};


int main() {
    PhoneBook phonebook;
    phonebook.add_subscriber("Mallaev", "+32395660114");
    phonebook.add_subscriber("Mallaevs", "+78062309422");
    phonebook.add_subscriber("Motygin", "+79802345252");
    phonebook.add_subscriber("Dronova", "+79190342372");
    phonebook.add_subscriber("Popov", "+9095824567234");
    phonebook.add_subscriber("Schevtsov", "+4589145880");
    phonebook.add_subscriber("Fakhrutdinov", "+91824619846");

    phonebook.find_for_typ("-1", "-1");
    std::cout << '\n';
    phonebook.find_for_typ("Mall", "Pop");
    std::cout << '\n';
    phonebook.find_random();
    std::cout << '\n';
    phonebook.find_quick("Mallaev");

    return 0;
}
