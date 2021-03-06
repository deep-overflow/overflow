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
    Linear(int in_features_, int out_features_, char init_ = 'h');
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
    Dropout(double ratio_);
    ~Dropout();

    virtual Tensor *operator()(Tensor *input_);

    virtual void backward();
    virtual void zero_grad();

    virtual void print();

    double ratio;
    int n_drop;
    Tensor *dropout;
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

class Softmax : public Function
{
public:
    Softmax();
    ~Softmax();

    virtual Tensor *operator()(Tensor *input_); // 구현한 식이 약간 헷갈림
    virtual void backward();                    // 구현한 식이 약간 헷갈림
    virtual void zero_grad();

    virtual void print();
};

class MSELoss : public Function
{
public:
    MSELoss();
    ~MSELoss();

    virtual Tensor *operator()(Tensor *output_, Tensor *label_);

    virtual void backward();
    virtual void zero_grad();

    virtual void print();
};

class CrossEntropyLoss : public Function
{
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    virtual Tensor *operator()(Tensor *output_, Tensor *label_);

    virtual void backward();
    virtual void zero_grad();

    virtual void print();

    Softmax softmax;
    Tensor *label;
};

#endif
