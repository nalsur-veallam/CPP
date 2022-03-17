# Answers on questions: #
## What should I think about before parallelizing a program? ##

When parallelizing, it is important to take into account not only the formal parallelism of the algorithm structure, but also the fact that exchange operations in parallel computers are, as a rule, much slower than arithmetic ones. The existence of the lion's share of the overhead costs for organizing parallelism is connected with this.

Parallel programming is used when for serial
program needs to reduce its execution time, or when it is sequential
the program, in view of the large amount of data, ceases to fit in the memory of one
computer.

## Name the main approaches to the organization of parallelism. ##

The development of parallel programs (PP) consists of three main stages:

* Decomposition of a task into subtasks. Ideally, these subtasks work independently
from each other (the principle of data locality). The exchange of data between subtasks is an expensive operation, especially if it is an exchange over a network.
* Distribution of a task among processors (virtual processors). In some cases this issue can be left up to the PP runtime environment.
* Writing a program using some parallel library. Choice libraries may depend on the platform on which the program will be executed, on the level of performance required and the nature of the task itself.

# Description of the written code #

* The for_each.cpp file contains code with an implemented parallel version of the std::for_each algorithm, using the std::async + std::future binding. The result of the test output of the program is recorded in the file for_each_out.txt. To run use
> g++ for_each.cpp -lpthread; ./a.out

* The accumulate.cpp file contains a modified parallel_accumulate algorithm so that the number of threads can be set externally. Using the timer.hpp header file, the program measures the running time of the algorithm for the number of threads in the range from 1 to 30. The result is output programmatically to a JSON file named time.json. Further, using the data from the JSON file, the graph.py script builds an graph.png graph, according to which we can conclude that for my AMD Ryzen 7-5700U processor with 16 threads on 8 cores, it makes sense to create up to 10 threads for this particular task. The result of the output of the C++ code is in the file output.txt. To run use
> g++ accumulate.cpp -lpthread; ./a.out
