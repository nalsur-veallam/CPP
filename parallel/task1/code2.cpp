#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <cassert>
#include <limits>
#include <mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<time.h>

#define K 10
int size = 700;

std::random_device randomizer;
std::mt19937 seed(randomizer());

const double lowRand = 0.0;
const double highRand = 1.0;

inline double getRandomReal(double low, double high)
{
    std::uniform_real_distribution<double> intDistrib(low, high);
    return intDistrib(seed);
}

void randomizeMatrix(double *A, int M, int N, double low, double high)
{
    for(int i = 0; i < M * N; ++i) {
        A[i] = getRandomReal(low, high);
    }
}

int main(int argc,char *argv[])
{
    double time = 0;
    int rank, numprocs;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
 
    int line = size/numprocs;
    
    double *a = new double[size*size];
    double *b = new double[size*size];
    double *c = new double[size*size];
    
    double *buffer = new double[size*line];
    double *ans = new double[size*line];
 
    // Основной процесс присваивает матрице начальное значение и передает матрицу N каждому процессу, а матрицу M передает каждому процессу в группах.
        if (rank==0)
        {
            randomizeMatrix(a, size, size, lowRand, highRand);
            randomizeMatrix(b, size, size, lowRand, highRand);
            std::fill(c, c + size*size, 0);
            
            auto start = std::chrono::steady_clock::now();
            // Отправить матрицу N другим подчиненным процессам
            for (int i=1;i<numprocs;i++)
            {
                    MPI_Send(b,size*size,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
            }
            // Отправляем каждую строку a каждому подчиненному процессу по очереди
            for (int l=1; l<numprocs; l++)
            {
                MPI_Send(a+(l-1)*line*size,size*line,MPI_DOUBLE,l,1,MPI_COMM_WORLD);
            }
            // Получаем результат, рассчитанный по процессу
            for (int k=1;k<numprocs;k++)
            {
                MPI_Recv(ans,line*size,MPI_DOUBLE,k,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                // Передаем результат в массив c
                for (int i=0;i<line;i++)
                {
                    for (int j=0;j<size;j++)
                    {
                        c[((k-1)*line+i)*size+j] = ans[i*size+j];
                    }
    
                }
            }
            // Рассчитать оставшиеся данные
            for (int i=(numprocs-1)*line;i<size;i++)
            {
                for (int j=0;j<size;j++)
                {
                    double temp=0;
                    for (int k=0;k<size;k++)
                        temp += a[i*size+k]*b[k*size+j];
                    c[i*size+j] = temp;
                }
            }
            
            auto end = std::chrono::steady_clock::now(); 
            std::chrono::duration<double> elapsed_seconds = end-start;
            time += double(elapsed_seconds.count());
            
            std::cout << numprocs << " " << 1000/K*time << std::endl;
    
            free(a);
            free(b);
            free(c);
            free(buffer);
            free(ans);
            
            
        }
    
        // Другие процессы получают данные и после вычисления результата отправляют их в основной процесс
        else
        {
            // Получаем широковещательные данные (матрица b)
            MPI_Recv(b,size*size,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
            MPI_Recv(buffer,size*line,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            // Рассчитать результат продукта и отправить результат в основной процесс
            for (int i=0;i<line;i++)
            {
                for (int j=0;j<size;j++)
                {
                    double temp=0;
                    for(int k=0;k<size;k++)
                        temp += buffer[i*size+k]*b[k*size+j];
                    ans[i*size+j]=temp;
                }
            }
            // Отправить результат расчета в основной процесс
            MPI_Send(ans,line*size,MPI_DOUBLE,0,3,MPI_COMM_WORLD);
        }
 
        MPI_Finalize();
 
    return 0;
}


// alpha (2000) 0.123
// alpha (500) 0.375
// alpha (700) 0.193
// total alpha 0.23

// Max S ~ 4.5

