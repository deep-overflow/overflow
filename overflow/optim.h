#ifndef __OPTIM_H__
#define __OPTIM_H__

#include <iostream>
#include "core.h"

class Optimizer;
class SGD;

class Optimizer
{
public:
    Optimizer();

    virtual void step();

    virtual void print();

    // variable
    Tensor **params;
    int num_params;

    std::string name;
};

class SGD : public Optimizer
{
public:
    SGD(Tensor **params_, int num_params_, double lr_ = 0.001, bool l2_reg_ = true);

    virtual void step();

    virtual void print();

    double lr;
    bool l2_reg;
};

#endif