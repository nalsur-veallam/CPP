#include <string>
#include <regex>
#include <iostream>
#include <filesystem>
#include <fstream>
#include "json.hpp"

using nlohmann::json;

class Person{
public:
    std::string name;
    int age;
    std::string DOB; //Date of birth
    std::string email;
    bool sex; // True if male and False if female
    int height;
    std::string city;
    
    void save(){
        json save;
        save["name"] = name;
        save["age"] = age;
        save["date_of_birth"] = DOB;
        save["email"] = email;
        save["sex"] = sex;
        save["height"] = height;
        save["city"] = city;
        
        auto path = std::filesystem::current_path();
        std::filesystem::create_directory(path / "savedata");
     
        
        std::fstream file;
        file.open("savedata/json" + name + ".txt", std::ios::trunc | std::ios::out);
        file << std::setw(6) << save;
    }
};

std::istream& operator>> (std::istream &in, Person &person)
{
    std::cout << "Enter a name and press Enter:\n";
    getline(in, person.name);
    std::cout << "Enter age and press Enter:\n";
    in >> person.age;
    int ch = 0;
    while ((ch = std::cin.get()) != '\n' && ch != EOF);
    bool do0 = true;
    while(do0){
        std::regex date_pattern(R"((0[1-9]|[12][0-9]|3[01])[- :/.](0[1-9]|1[012])[- :/\.]([012]\d{3}))");
        std::cout << "Enter date of birth in the format DD:MM:YYYY and press Enter:\n";
        getline(in, person.DOB);
        if(std::regex_match(person.DOB, date_pattern) ){ do0 = false;}
        else {std::cout << "Error!\n";}
    }
    do0 = true;
    while(do0){
        std::regex email_pattern("(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+");
        std::cout << "Enter email in the format example@example.example and press Enter:\n";
        getline(in, person.email);
        if(std::regex_match(person.email, email_pattern)) { do0 = false;}
        else {std::cout << "Error!\n";}
    }
    std::cout << "If" << person.name << "'s sex is male, write m, otherwise f and press Enter at the end\n";
    do0 = true;
    while(do0){
        char s;
        in >> s;
        if (!std::cin.good()) {
        std::cin.clear();
        int ch = 0;
        while ((ch = std::cin.get()) != '\n' && ch != EOF);
        }
        else{
            if (s == 'm'){person.sex = true; do0 = false;}
            else if(s == 'f'){person.sex = false; do0 = false;}
            else {std::cout << "Error!; If" << person.name << "'s sex is male, write m, otherwise f and press Enter at the end\n";}
        }
    }
    ch = 0;
    while ((ch = std::cin.get()) != '\n' && ch != EOF);
    std::cout << "Enter name of city and press Enter:\n";
    getline(in, person.city);
    std::cout << "Enter height and press Enter:\n";
    in >> person.height;
    ch = 0;
    while ((ch = std::cin.get()) != '\n' && ch != EOF);
    return in;
}

int main(){
    Person max;
    std::cin >> max;
    max.save();
    Person me;
    std::cin >> me;
    me.save();
    return 0;    
}






