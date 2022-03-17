#include <iostream>
#include <chrono>

template <typename T>
class Timer {
private:
    using clock_t = std::chrono::steady_clock;
    using time_point_t = clock_t::time_point;
    using Time = T;

    time_point_t start, pause_time;
    Time time{};

public:

    Timer() {
        time = Time(0);
        start = clock_t::now();
    }

    void start_timer() {
        start = clock_t::now();
    };

    void pause_timer() {
        time += std::chrono::duration_cast<Time>(clock_t::now() - start);
        pause_time = std::chrono::steady_clock::now();
    }

    void resume_timer(){
        start += std::chrono::steady_clock::now() - pause_time;
    }

    void reset_timer() {
        time = Time(0);
        start = clock_t::now();
    }

    void print_time() {
        this->pause_timer();
        std::cout << "Current time: " << time.count() << " milliseconds\n";
    }

    long int return_time(){
        this->pause_timer();
        return time.count();
    }
};
