#include "optim.h"

// Optimizer ##############################################

Optimizer::Optimizer() : params(NULL), num_params(0), verbose(false)
{
    if (verbose)
        std::cout << "Optimizer::Optimizer()" << std::endl;

    name = "< Optimizer class >";
}

void Optimizer::step()
{
    if (verbose)
        std::cout << "void Optimizer::step()" << std::endl;

    std::cerr << "Not Implemented Error" << std::endl;
}

void Optimizer::print()
{
    std::cout << name << std::endl;

    std::cerr << "Not Implemented Error" << std::endl;
}

// SGD ####################################################

SGD::SGD(Tensor **params_, int num_params_, double lr_, bool l2_reg_) : lr(lr_), l2_reg(l2_reg_)
{
    if (verbose)
        std::cout << "SGD::SGD(Tensor **params_, int num_params_, double lr_, bool l2_reg_)" << std::endl;

    params = params_;
    num_params = num_params_;
    name = "< SGD class : Optimizer class>";
}

void SGD::step()
{
    if (verbose)
        std::cout << "void SGD::step()" << std::endl;

    Tensor *param;
    for (int i = 0; i < num_params; i++)
    {
        param = params[i];

        for (int j = 0; j < param->tensor_shape.size; j++)
        {
            double grad = param->grad[j];
            
            if (l2_reg)
            {
                grad += param->data[j];
            }

            param->data[j] -= lr * grad;
        }
    }
}

void SGD::print()
{
    std::cout << name << std::endl;

    std::cout << "lr : " << lr << std::endl;

    if (l2_reg)
    {
        std::cout << "l2_reg : true" << std::endl;
    }
    else
    {
        std::cout << "l2_reg : false" << std::endl;
    }

    std::cout << "num_params : " << num_params << std::endl;

    for (int i = 0; i < num_params; i++)
    {
        std::cout << i + 1 << "th param :" << std::endl;
        params[i]->print();
    }
}

// Linear #################################################

Adam::Adam(Tensor **params_, int num_params_, double lr_, double beta1_, double beta2_,
           double eps_, double weight_decay_, bool amsgrad_, bool maximize_)
    : lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_), weight_decay(weight_decay_), amsgrad(amsgrad_), maximize(maximize_)
{
    if (verbose)
        std::cout << "Adam::Adam(Tensor **params_, int num_params_, double lr_ = 0.001, double beta1_ = 0.9, double beta2_ = 0.999, double eps_ = 1e-8, double weight_decay_ = 0, bool amsgrad_ = false, bool maximize_ = false)" << std::endl;

    params = params_;
    num_params = num_params_;
    name = "< Adam class : Optimizer class >";
}

void Adam::step()
{
    if (verbose)
        std::cout << "void Adam::step()" << std::endl;
}

void Adam::print()
{
    std::cout << name << std::endl;
    std::cout << "lr : " << lr << std::endl;
    std::cout << "beta1 : " << beta1 << std::endl;
    std::cout << "beta2 : " << beta2 << std::endl;
    std::cout << "eps : " << eps << std::endl;
    std::cout << "weight_decay : " << weight_decay << std::endl;

    if (amsgrad)
        std::cout << "amsgrad : true" << std::endl;
    else
        std::cout << "amsgrad : false" << std::endl;

    if (maximize)
        std::cout << "maximize : true" << std::endl;
    else
        std::cout << "maximize : false" << std::endl;
}