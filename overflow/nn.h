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
    Linear(int in_features_, int out_features_);

    virtual Tensor *operator()(Tensor *input_);

    virtual void backward();

    virtual void print();

    Tensor params;
};

class ReLU
{
public:
};

class MSELoss
{
public:
};

#endif