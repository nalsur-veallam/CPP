CPP=mpiCC
FLAGS=-O3

N=2000

all : matrix

matrix : bin/start

clean :
	- rm -r bin
	- rm -r obj
	

check :
	./bin/start $N s
	#./bin/start $N m
	
obj :
	mkdir obj

bin :
	mkdir bin
	
obj/matrix.o: src/matrix.cpp | obj
	$(CPP) -c src/matrix.cpp -o obj/matrix.o $(FLAGS)
obj/main.o: src/main.cpp | obj
	$(CPP) -c src/main.cpp -o obj/main.o $(FLAGS)
bin/start: obj/matrix.o obj/main.o | bin
	$(CPP) obj/main.o obj/matrix.o -o bin/start $(FLAGS)
