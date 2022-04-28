#include "optim.h"

// Optimizer ##############################################

Optimizer::Optimizer() : params(NULL), num_params(0)
{
    std::cout << "Optimizer::Optimizer()" << std::endl;

    name = "< Optimizer class >";
}

void Optimizer::step()
{
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
    std::cout << "SGD::SGD(Tensor **params_, int num_params_, double lr_, bool l2_reg_)" << std::endl;

    params = params_;
    num_params = num_params_;
    name = "< SGD class : Optimizer class>";
}

void SGD::step()
{
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