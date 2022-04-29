#ifndef __NN_H__
#define __NN_H__

#include <iostream>
#include "core.h"

class Linear;
class ReLU;
class MSELoss;

class Linear : public Function
{
public:
    Linear(int in_features_, int out_features_, char init_ = 'n');
    ~Linear();

    virtual Tensor *operator()(Tensor *input_);

    virtual void backward();
    virtual void zero_grad();

    virtual Tensor *return_params();

    virtual void print();

    Tensor params;
    char init;
};

class Dropout : public Function
{
public:
    Dropout();
    ~Dropout();

    virtual Tensor *operator()(Tensor *input_);

    virtual void backward();
    virtual void zero_grad();

    virtual Tensor *return_params();

    virtual void print();
};

class ReLU : public Function
{
public:
    ReLU();
    ~ReLU();

    virtual Tensor *operator()(Tensor *input_);

    virtual void backward();
    virtual void zero_grad();

    virtual void print();
};

class Sigmoid : public Function
{
public:
    Sigmoid();
    ~Sigmoid();

    virtual Tensor *operator()(Tensor *input_);
    virtual void backward();
    virtual void zero_grad();

    virtual void print();
};

class MSELoss : public Function
{
public:
    MSELoss();
    ~MSELoss();

    virtual Tensor *operator()(Tensor *input_1, Tensor *input_2);

    virtual void backward();
    virtual void zero_grad();

    virtual void print();
};

#endif