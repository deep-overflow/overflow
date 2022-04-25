#ifndef __CORE_H__
#define __CORE_H__

#include <iostream>
#include <string>
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
    Shape(const Shape &a);

    void operator=(const Shape &a);
    bool operator==(const Shape &a) const;
    bool operator!=(const Shape &a) const;

    void reshape(const int *shape_, const int dim_);
    void reshape(const Shape &a);
    void T();
    Shape index(int s, int e) const;

    void print();

    // ===== variable =====
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
    Tensor(const int *shape_, const int dim_);
    // Tensor(const double *data_, const Shape &shape_);
    Tensor(const double data_, const Shape &shape_);
    Tensor(const Shape &shape_);
    ~Tensor();

    void operator=(const Tensor &a);
    Tensor operator+(const Tensor &a);
    Tensor operator-(const Tensor &a);
    Tensor operator^(const Tensor &a);
    Tensor operator^(int k);
    // void operator*(const Tensor& c);

    void init(const double data_, const int *shape_, const int dim_);
    void init_like(const double data_, const Shape& shape_);
    void init_like(const double data_, const Tensor& a);
    void random(const int *shape_, const int dim_, char init_='n');
    void random(const Shape &shape_, char init_='n');
    void random(char init_='n');

    Tensor index(int arg_num, ...) const; // not general 수정 필요.
    double index_(int arg, ...) const;
    double grad_index(int arg, ...) const;

    double sum_(int axis = -1, ...); // 수정해야 됨.
    Tensor sum(int axis = -1); // 수정해야 됨.
    void append(const Tensor &a, bool new_axis = true);

    void backward();
    void zero_grad();

    void dot(const Tensor &a); // not generalized: for matrix
    void grad_dot(const Tensor &a); // not generalized: for matrix
    void T(); // not generalized: for matrix

    void print();

    // ===== variable =====
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
    std::string name;
};

#endif