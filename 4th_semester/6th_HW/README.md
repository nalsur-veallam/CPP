# Answers on questions: #
## What should be paid attention to when parallelizing programs? ##

Sequential area - one process (thread), entry into the parallel area - the generation of a certain number of processes, their completion, the master thread. Parallel regions can be nested within each other.

Unlike full-fledged processes, spawning threads is a relatively fast operation, so frequent spawns and terminations of threads do not affect program execution time as much.

To write an efficient parallel program, it is necessary that all threads be evenly loaded, which is achieved by careful load balancing.

The need to synchronize access to shared data is essential. The very existence of data that is shared by multiple threads leads to conflicts with simultaneous inconsistent access.

The user must explicitly use synchronization directives or appropriate library functions. When each thread accesses its own file, no synchronization is required.

## What is an atomic operation and an atomic data type?. ##

* An operation on a shared memory area is said to be atomic if it completes in one step relative to other threads accessing that memory. While such an operation is being performed on a variable, no thread can see the change half-completed. Atomic loading ensures that the entire variable is loaded at one point in time. Non-atomic operations provide no such guarantee.

* Atomic data. In database theory, these are attributes that store a single value and are neither a list nor a set of values. In other words, these are data, the division of which into components leads to the loss of their meaning from the point of view of the problem being solved. For example, if the attribute "Price" contains the value 15, then trying to divide it by 1 and 5 will lead to complete nonsense.

-------------------------------------------------------------------------------------------

The main sources of multithreading problems are data races and race conditions. In simple terms, C++ defines data races as simultaneous and unsynchronized accesses to the same memory location, with one of the accesses modifying the data. Whereas race conditions is a more general term that describes a situation where the result of program execution depends on the sequence or timing of threads.

The main problem with race conditions is that they may not be noticeable during software development and may disappear during debugging. This behavior often leads to a situation where the application is considered complete and correct, but the end user experiences intermittent problems, often of an obscure nature. To solve the data race problem, C++ provides a set of interfaces such as atomic operations and primitives for creating critical sections (mutexes).

Atomic operations are a powerful tool that allows you to avoid data races and create efficient synchronization algorithms. However, this creates a convoluted C++ memory model, which introduces yet another layer of complexity.

Using a mutex is often much easier than using atomic operations. It allows you to create a critical section that can be executed by at most one thread at any given time. In addition, advanced mutexes such as shared_lock can in some cases improve efficiency by allowing a group of threads to execute a critical section if the mutex is not locked into exclusive use.

# Description of the written code #

* The file ts_queue.hpp contains a thread-safe queue class in C++, which is tested by the code in main.cpp. This code calculates the program's running time depending on the number of tasks (just for the sake of interest, I did it). The result of the program is in the file out.txt. A graph was also built by the script graph.py based on the received data. The graph is in the file graph.pdf. To run use:
> g++ main.cpp -lpthread -std=c++17; ./a.out
