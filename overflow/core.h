#ifndef __CORE_H__
#define __CORE_H__

#include <iostream>
#include <random>
#include <cstdarg>

class Module;
class Shape;
class Tensor;
class Function;

class Module
{
public:
    Module();

    Tensor *operator()(Tensor *input_);

    void add_params();

    void print();

    Function **func;
    Tensor **params;
    int n_func;
    int n_params;
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
    void random_init(const int *shape_, const int dim_, char init_);
    void random_init(char init_);
    // void init(); // randomized initialization

    double sum(int axis = -1);
    void append(const Tensor &a);
    Tensor index_(int arg, ...) const; // 구현해야 함.
    double index(int arg, ...) const; // general
    double grad_index(int arg, ...) const; // general

    void backward();
    void zero_grad();

    void dot(const Tensor &a);
    void grad_dot(const Tensor &a);
    void T();

    void print(); // general

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

    virtual Tensor *return_params();

    virtual void print();

    // variable
    Tensor *input;
    Tensor *input2;
    Tensor *output;
    bool has_params;
};

#endif