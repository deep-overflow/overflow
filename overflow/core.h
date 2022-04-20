#ifndef __CORE_H__
#define __CORE_H__

#include <iostream>
#include <random>
#include <cstdarg>

class Axis;
class Shape;
class Tensor;
class Function;

class Axis
{
public:
    Axis(int start_, int end_, int stride_) : start(start_), end(end_), stride(stride_) {}
    int start;
    int end;
    int stride;
    int n_elements;
};

class Shape
{
public:
    Shape();
    Shape(const int *shape_, const int dim_);

    void operator=(const Shape &a);
    bool operator==(const Shape &a);

    void init(const int *shape_, const int dim_);

    void T();

    void print();

    // variable
    int *shape;
    int dim;
    int size;
};

class Tensor
{
public:
    Tensor();
    Tensor(const double *data_, const int *shape_, const int dim_);
    Tensor(const double data_, const int *shape_, const int dim_);
    Tensor(const double data_, const Shape &shape_);
    ~Tensor();

    void operator=(const Tensor &a);
    Tensor operator+(const Tensor &a);
    Tensor operator-(const Tensor &a);
    Tensor operator^(int k);
    // void operator*(const Tensor& c);

    void init(const double data_, const int *shape_, const int dim_);
    // void random_init(const int *shape_, const int dim_, char init_);
    void random_init(char init_);
    // void init(); // randomized initialization

    double sum(int axis = -1);
    double index(int i, int j) const; // for matrix

    // Tensor indexing(int dim_, ...) const;  // general
    double grad_index(int i, int j) const; // for matrix

    void backward();
    void zero_grad();

    void dot(const Tensor &a);
    void grad_dot(const Tensor &a);
    void T();

    void print();

    // variable
    double *data;
    double *grad;
    Shape tensor_shape;
    bool requires_grad;

    Function *func;
};

Tensor dot(const Tensor &a, const Tensor &b);

class Function
{
public:
    Function();

    virtual Tensor *operator()(Tensor *input_);
    virtual Tensor *operator()(Tensor *input_1, Tensor *input_2);

    virtual void backward();
    virtual void zero_grad();

    virtual void print();

    // variable
    Tensor *input;
    Tensor *input2;
    Tensor *output;
};

#endif