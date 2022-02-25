
Here the code of the third homework is implemented (various work with arrays and processing of their data). The files output.txt and image.png contain an example of the program.



Answers on questions:

1) A pseudo-random number generator is a program that takes a start/seed value and performs certain mathematical operations on it to convert it to another number that is not related to the start value at all. The program then uses the newly generated value and performs the same mathematical operations on it as it did on the seed to convert it to another new number, a third, which is not related to either the first or the second. By applying this algorithm to the last generated value, the program can generate a whole series of new numbers that will appear to be random (assuming the algorithm is sufficiently complex).

C++11 added a ton of new functionality for generating random numbers, including the Mersenne Twist algorithm, as well as different types of random number generators (for example, uniform, Poisson generator, etc.). The Mersenne Twist generates random 32-bit unsigned integers (rather than 15-bit integers as is the case with rand()), which allows for a much larger range of values. There is also a version (std::mt19937_64) for generating 64-bit unsigned integers.

To prevent the sequence of random numbers from repeating every time you start the program, set the "seed" of the pseudo-random generator as the current time or, in the case of some retro (and not only) games, the intervals between keystrokes from the keyboard / joystick.

Distribution is a law that describes the range of values of a random variable.

2) An iterator is an object designed specifically to iterate over the elements of a container (for example, the values of an array or the characters in a string), providing access to each of them while iterating through the elements.

The programmer can use the interface provided by this iterator to move through or access the elements of the container without worrying about what type of element iteration is involved or how the container stores data.

A range is a sequence of elements stored sequentially between two iterators, called start and end points.
