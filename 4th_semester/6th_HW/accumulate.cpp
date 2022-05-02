#include <thread>
#include <numeric>
#include <iostream>
#include <vector>
#include <fstream>
#include <atomic>
#include <algorithm>

template <typename Iterator, typename T>
void accumulate_block(Iterator first, Iterator last, T init, std::atomic<T>& result) {
    result = std::accumulate(first, last, init);
}

template<typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init) {
    // 1. Проверили длину
    auto num_threads = std::thread::hardware_concurrency();
    auto length = distance(first, last);
    if (length < 4 * num_threads) {
        return std::accumulate(first, last, init);
    }
    // 2. Длина достаточна, распараллеливаем
    // Вычислили длину для одного потока
    auto length_per_thread = length / num_threads;
    // Векторы с потоками и результатами
    std::vector <std::thread> threads;
    threads.reserve(num_threads - 1);
    std::atomic<T> result = init;
    // 3. Распределяем данные (концепция полуинтервалов!)
    auto beginning = first;
    auto ending = std::next(first, length_per_thread);
    for (int i = 0; i < num_threads - 1; i++) {
        beginning = std::next(first, i * length_per_thread);
        ending = std::next(first, (i + 1) * length_per_thread);
        // 4. Запускаем исполнителей
        threads.push_back(std::thread(accumulate_block<Iterator, T>, beginning, ending, 0, std::ref(result)));
    }
    result += std::accumulate(std::next(first, (num_threads - 1) * length_per_thread), last, 0);
    // std::mem_fun_ref -- для оборачивания join().
    std::for_each(std::begin(threads), std::end(threads), std::mem_fun_ref(&std::thread::join));
    // 5. Собираем результаты
    return result;
}

int main() {
    std::vector<int> test_sequence(1e9);
    std::iota(test_sequence.begin(), test_sequence.end(), 0);
    std::cout << ". Result is " << parallel_accumulate(std::begin(test_sequence),
                                std::end(test_sequence), 0) << std::endl;
    return 0;
}
