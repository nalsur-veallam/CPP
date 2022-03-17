#include <thread>
#include <numeric>
#include <iostream>
#include <vector>
#include <fstream>
#include "json.hpp"
#include "timer.hpp"

using nlohmann::json;

template <typename Iterator, typename T>
void accumulate_block(Iterator first, Iterator last, T init, T& result) {
    result = std::accumulate(first, last, init);
}

template<typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init, unsigned num_threads) {
    // 1. Проверили длину
    auto length = distance(first, last);
    if (length < 4 * num_threads) {
        return std::accumulate(first, last, init);
    }
    // 2. Длина достаточна, распараллеливаем
    // Вычислили длину для одного потока
    auto length_per_thread = (length + num_threads - 1) / num_threads;
    // Векторы с потоками и результатами
    std::vector <std::thread> threads;
    std::vector <T> results(num_threads - 1);
    // 3. Распределяем данные (концепция полуинтервалов!)
    auto beginning = first;
    auto ending = std::next(first, length_per_thread);
    for (int i = 0; i < num_threads - 1; i++) {
        beginning = std::min(std::next(first, i * length_per_thread), last);
        ending = std::min(std::next(first, (i + 1) * length_per_thread), last);
        // 4. Запускаем исполнителей
        threads.push_back(std::thread(accumulate_block<Iterator, T>, beginning, ending, 0, std::ref(results[i])));
    }
    // Остаток данных -- в родительском потоке
    auto main_result = std::accumulate(std::min(std::next(first, (num_threads - 1) * length_per_thread), last), last, init);
    // std::mem_fun_ref -- для оборачивания join().
    std::for_each(std::begin(threads), std::end(threads), std::mem_fun_ref(&std::thread::join));
    // 5. Собираем результаты
    return accumulate(std::begin(results), std::end(results), main_result);
}

int main() {
    std::vector <long int> time;
    Timer <std::chrono::milliseconds> t;
    std::vector<int> test_sequence(1e9);
    for(auto i = 1; i <= 30; i++){
        t.reset_timer();
        std::iota(test_sequence.begin(), test_sequence.end(), i);
        auto result =
                parallel_accumulate(std::begin(test_sequence),
                                    std::end(test_sequence), 0, i);
        std::cout << i << ". Result is " << result << std::endl;
        time.push_back(t.return_time());
    }
    json save;
    save["time"] = time;    
    
    std::fstream file;
    file.open("time.json", std::ios::trunc | std::ios::out);
    file << std::setw(0) << save;
    return 0;
}
/*

int main() {
    std::vector<int> test_sequence(100u);
    std::iota(test_sequence.begin(), test_sequence.end(), 0);
    auto result =
            parallel_accumulate(std::begin(test_sequence),
                                std::end(test_sequence), 0);
    std::cout << "Result is " << result << std::endl;
}*/

