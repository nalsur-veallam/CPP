#include <iostream>
#include <thread>
#include <atomic>
#include "ts_queue.hpp"
#include "timer.hpp"

class Threads_Guard
{
public:

    explicit Threads_Guard(std::vector < std::thread > & threads) :
            m_threads(threads)
    {}

    Threads_Guard			(Threads_Guard const&) = delete;

    Threads_Guard& operator=(Threads_Guard const&) = delete;

    ~Threads_Guard() noexcept
    {
        try
        {
            for (auto & m_thread : m_threads)
            {
                if (m_thread.joinable())
                {
                    m_thread.join();
                }
            }
        }
        catch (...)
        {
            // std::abort();
        }
    }

private:

    std::vector < std::thread > & m_threads;
};

template <typename Queue>
void pushQueue(Queue& queue, int number_to_push, std::atomic<bool>& flag) {

    while (!flag.load()) {
        std::this_thread::yield();
    }

    for (int i = 0; i < number_to_push; i++) {
        queue.push(i);
    }

}

void popOurQueue(Threadsafe_Queue<int>& queue, int number_to_pop, std::atomic<bool>& flag) {

    while (!flag.load()) {
        std::this_thread::yield();
    }

    int res;
    for (int i = 0; i < number_to_pop; i++) {
        queue.wait_and_pop(res);
    }
}

void testQueue() {
    const int number_threads = 8;

    for(int number_tasks = 1000; number_tasks <= 1e5; number_tasks += 1000){
        Threadsafe_Queue<int> queue;

        Timer <std::chrono::milliseconds> t;

        {
            std::atomic<bool> flag(false);

            std::vector<std::thread> threads(2 * number_threads);
            Threads_Guard guard(threads);

            for (int i = 0; i < 2 * number_threads; i++) {
                if (i < number_threads) {

                    threads[i] = std::thread(pushQueue<Threadsafe_Queue<int>>,
                                            std::ref(queue), number_tasks, std::ref(flag));
                }
                else {

                    threads[i] = std::thread(popOurQueue, std::ref(queue), number_tasks, std::ref(flag));
                }
            }

            t.start_timer();
            flag.store(true);

        }
        std::cout << "Time for "<< number_tasks << " tasks - " << t.return_time() << " milliseconds";

        std::cout << "\n";

        t.reset_timer();
    }
}

int main()
{
    testQueue();
    return 0;
}
