#include "optim.h"

// Optimizer ##############################################

Optimizer::Optimizer() : params(NULL), num_params(0)
{
    std::cout << "Optimizer::Optimizer()" << std::endl;
}

void Optimizer::step()
{
    std::cout << "void Optimizer::step()" << std::endl;

    std::cerr << "Not Implemented Error" << std::endl;
}

// SGD ####################################################

SGD::SGD(Tensor **params_, int num_params_, double lr_, bool l2_reg_) : lr(lr_), l2_reg(l2_reg)
{
    std::cout << "SGD::SGD(Tensor **params_, int num_params_, double lr_, bool l2_reg_)" << std::endl;

    params = params_;
    num_params = num_params_;
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
