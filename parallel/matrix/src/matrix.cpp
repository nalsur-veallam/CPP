#include <iostream>
#include <vector>
#include <fstream>
#include "math.h"
#include "matrix.hpp"

using namespace std;

double matrix::GetIJComp(unsigned int i_, unsigned int j_) const{
	return Matrix[i_][j_];
}

void matrix::SetIJComp(double x_, unsigned int i_, unsigned int j_){
	 Matrix[i_][j_] = x_;
}

unsigned int matrix::GetN() const { return n; }

unsigned int matrix::GetM() const { return m; }


matrix ::matrix(unsigned int n_)
{

    this->n = n_;
    this->m = n_;
    Matrix.resize(n_);

    for (unsigned int i = 0; i < n_; i++)
    {
        Matrix[i].resize(n_);
    }

    for (unsigned int i = 0; i < n_; i++)
    {
        for (unsigned int j = 0; j < n_; j++)
        {
            Matrix[i][j] = 0;
        }
    }
}

inline double getRandomReal(double low, double high)
{
    std::uniform_real_distribution<double> intDistrib(low, high);
    return intDistrib(seed);
}

matrix ::matrix(unsigned int n_, unsigned int m_, double low, double high)
{
    this->n = n_;
    this->m = n_;
    Matrix.resize(n_);

    for (unsigned int i = 0; i < n_; i++)
    {
        Matrix[i].resize(n_);
    }

    for (unsigned int i = 0; i < n_; i++)
    {
        for (unsigned int j = 0; j < n_; j++)
        {
            Matrix[i][j] = getRandomReal(low, high);
        }
    }
}

matrix ::matrix(unsigned int n_, unsigned int m_)
{

    this->n = n_;
    this->m = m_;
    Matrix.resize(n_);

    for (unsigned int i = 0; i < n_; i++)
    {
        Matrix[i].resize(m_);
    }

    for (unsigned int i = 0; i < n_; i++)
    {
        for (unsigned int j = 0; j < m_; j++)
        {
            Matrix[i][j] = 0;
        }
    }
}

matrix ::matrix(const matrix &rhs)
{

    n = rhs.GetN();
    m = rhs.GetM();

    Matrix.resize(n);

    for (unsigned int i = 0; i < n; i++)
    {
        Matrix[i].resize(n);
    }

    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < m; j++)
        {
            Matrix[i][j] = rhs.Matrix[i][j];
        }
    }
}

matrix ::~matrix(){};

matrix matrix::operator+(matrix const &rhs)
{
    unsigned int n_ = n;
    unsigned int m_ = m;
    matrix AddMatrix(n_, m_);
    
    for (unsigned int i = 0; i < n_; i++)
    {
        for (unsigned int j = 0; j < m_; j++)
        {
            AddMatrix.SetIJComp(rhs.GetIJComp(i,j) + Matrix[i][j], i, j);
        }
    }
    
    return AddMatrix;
}

matrix matrix::operator-(matrix const &rhs)
{
    unsigned int n_ = n;
    unsigned int m_ = m;
    matrix DiffMatrix(n_, m_);
    
    for (unsigned int i = 0; i < n_; i++)
    {
        for (unsigned int j = 0; j < m_; j++)
        {
            DiffMatrix.SetIJComp(Matrix[i][j] - rhs.GetIJComp(i,j), i, j);
        }
    }
    
    return DiffMatrix;
}

matrix matrix::operator*(matrix const &rhs) const
{
    if (m == rhs.GetN())
    {
        unsigned int n_ = n;
        unsigned int m_ = rhs.GetM();
        matrix MultMatrix(n_, m_);
        
        for (unsigned int i = 0; i < n_; i++)
        {
            for (unsigned int k = 0; k < m; k++)
            {
                for (unsigned int j = 0; j < m_; j++)
                {
                    MultMatrix.SetIJComp(MultMatrix.GetIJComp(i,j) + this->GetIJComp(i, k) * rhs.GetIJComp(k,j), i, j);
                }
            }
        }
        
        return MultMatrix;
    }
    else {cout << "\n Error: Inappropriate matrix sizes for multiplication"; return 0;}
}

matrix matrix ::t()
{
    unsigned int n_ = m;
    unsigned int m_ = n;
    matrix T(n_, m_);
    
    for (unsigned int i = 0; i < m_; i++)
    {
        for (unsigned int j = 0; j < n_; j++)
        {
            T.SetIJComp(this->GetIJComp(j,i), i, j);
        }
    }
    
    return T;
}

matrix matrix ::inv()
{
    if (n != m) {cout << "\n Error: The matrix is not square"; return 0;}
    
    unsigned int n_ = n;
    matrix invm(n_);
    matrix extended(n_, 2 * n_);

    for (unsigned int i = 0; i < n_; i++)
    {
        for (unsigned int j = 0; j < 2 * n_; j++)
        {
            if (j < n)
            {
                extended.SetIJComp(Matrix[i][j], i, j);
            }
            else if (j >= n)
            {
                if (i == (j - n))
                {
                    extended.SetIJComp(1, i, j);
                }
            }
        }
    }
    double extended_ii_max = 0;
    unsigned int i_max = 0;
    matrix leading_line(2 * n_, 1);

    for (unsigned int i = 0; i < n_; i++)
    {
        i_max = i;
        extended_ii_max = fabs(extended.GetIJComp(i, i));
        for (unsigned int k = i; k < n_; k++)
        {
            if (extended_ii_max <= fabs(extended.GetIJComp(k, i)))
            {
                extended_ii_max = fabs(extended.GetIJComp(k, i));
                i_max = k;
            }
        }
        if (extended_ii_max == 0)
        {
            cout << "Zero determinant exception" << endl;
            break;
        }
        for (unsigned int j = 0; j < 2 * n_; j++)
        {
            leading_line.SetIJComp(extended.GetIJComp(i, j), j, 0);
        }
        for (unsigned int j = 0; j < 2 * n_; j++)
        {
            extended.SetIJComp(extended.GetIJComp(i_max, j), i, j);
            extended.SetIJComp(leading_line.GetIJComp(j, 0), i_max, j);
        }
        for (unsigned int k = i + 1; k < n_; k++)
        {
            double f = extended.GetIJComp(k, i) / extended.GetIJComp(i, i);
            for (unsigned int j = 0; j < 2 * n_; j++)
            {
                extended.SetIJComp(extended.GetIJComp(k, j) - extended.GetIJComp(i, j) * f, k, j);
            }
        }
    }

    for (unsigned int i = 0; i < n_; i++)
    {
        double f = extended.GetIJComp(i, i);
        for (unsigned int j = 0; j < 2 * n_; j++)
        {
            extended.SetIJComp(extended.GetIJComp(i, j) / f, i, j);
        }
    }

    for (unsigned int i = n_ - 1; i >= 0; i--)
    {
        for (unsigned int k = i - 1; k >= 0; k--)
        {
            double f = extended.GetIJComp(k, i);
            for (unsigned int j = 0; j < 2 * n_; j++)
            {
                extended.SetIJComp(extended.GetIJComp(k, j) - extended.GetIJComp(i, j) * f, k, j);
            }
        }
    }

    for (unsigned int i = 0; i < n_; i++)
    {
        for (unsigned int j = n_; j < 2 * n_; j++)
        {
            invm.SetIJComp(extended.GetIJComp(i, j), i, j - n);
        }
    }

    return invm;
}

// overload << operator

ostream &operator<<(ostream &stream, matrix &mat)
{
    unsigned int n_ = mat.n;
    unsigned int m_ = mat.m;

    for (unsigned int i = 0; i < n_; i++)
    {
        for (unsigned int j = 0; j < m_; j++)
        {
            stream << mat.Matrix[i][j] << "  ";
        }
        stream << endl;
        stream << endl;
    }
    stream << endl;
    return stream;
}

