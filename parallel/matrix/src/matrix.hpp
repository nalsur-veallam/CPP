#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include "math.h"
#include "mpi.h"
using namespace std;

class matrix
{
public:
    double GetIJComp(unsigned int i_, unsigned int j_) const;
    void SetIJComp(double x_, unsigned int i_, unsigned int j_);
    unsigned int GetN() const;
    unsigned int GetM() const;
    matrix operator+(const matrix &);       // overload operator +
    matrix operator*(const matrix &) const; // overload operator *
    matrix operator-(const matrix &);       // overload operator -
    matrix operator^(const matrix &) const;
    matrix t();                             // transposed matrix
    matrix inv();                           // inverse matrix
    matrix(unsigned int n_, unsigned int m_, double low, double high);
    matrix(const matrix &);
    matrix(unsigned int n_);
    matrix(unsigned int n_, unsigned int m_);
    ~matrix();
    friend ostream &operator<<(ostream &stream, matrix &mat); // overload << operator
    
    

private:
    unsigned int n;
    unsigned int m;
    std::vector<std::vector<double>> Matrix;
};
