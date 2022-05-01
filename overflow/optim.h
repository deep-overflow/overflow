#ifndef __OPTIM_H__
#define __OPTIM_H__

#include <iostream>
#include "core.h"

class Optimizer;
class SGD;
class Adam;

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

    bool verbose;
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

class Adam : public Optimizer
{
public:
    Adam(Tensor **params_, int num_params_, double lr_ = 0.001, double beta1_ = 0.9, double beta2_ = 0.999,
         double eps_ = 1e-8, double weight_decay_ = 0, bool amsgrad_ = false, bool maximize_ = false);

    virtual void step();

    virtual void print();

    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    bool amsgrad;
    bool maximize;
};

#endif