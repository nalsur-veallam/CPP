#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <thread>

template <typename T>
class Timer {
private:
    using clock_t = std::chrono::steady_clock;
    using time_point_t = clock_t::time_point;

    time_point_t start, pause_time;
    T time{};

public:

    Timer() {
        time = T(0);
        start = clock_t::now();
    }

    void start_timer() {
        start = clock_t::now();
    };

    void pause_timer() {
        pause_time = std::chrono::steady_clock::now();
    }

    void resume_timer(){
        start += std::chrono::steady_clock::now() - pause_time;
    }

    void reset_timer() {
        time = T(0);
    }

    void print_time() {
        std::cout << "Current time: " << time.count() << " milliseconds\n";
    }

    void delay(int del){
        std::this_thread::sleep_for(std::chrono::milliseconds(del));
        time = T(del + time.count());
    }
};

int main() {
    // Test timer
    Timer <std::chrono::milliseconds> t;

    t.start_timer();

    std::cout << "Time is displayed once per second. Then the time stops for 2 seconds, and then it is displayed on the screen every 1.5 seconds\n";

    for(int i = 0; i < 10; i++){
        t.delay(1000);
        t.print_time();
    }
    t.pause_timer();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    t.resume_timer();
    for(int i = 0; i < 10; i++){
        t.delay(1500);
        t.print_time();
    }

    return 0;
}
