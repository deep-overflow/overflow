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

    // variable
    Tensor **params;
    int num_params;
};

class SGD : Optimizer
{
public:
    SGD(Tensor **params_, int num_params_, double lr_ = 0.001);

    virtual void step();

    double lr;
};

#endif