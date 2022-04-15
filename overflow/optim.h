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

    // variable
    Tensor *params;
};

class SGD : Optimizer
{
public:
    SGD(double lr = 0.001);
};

#endif