#include "optim.h"

// Optimizer ##############################################

Optimizer::Optimizer()
{
    params = NULL;
    num_params = 0;
}

void Optimizer::step()
{
    std::cerr << "Not Implemented Error" << std::endl;
}

// SGD ####################################################

SGD::SGD(Tensor *params_, int num_params_, double lr_)
{
    params = params_;
    num_params = num_params_;
    lr = lr_;
}

void SGD::step()
{
    Tensor *param;
    for (int i = 0; i < num_params; i++)
    {
        param = &params[i];
        
        for (int j = 0; j < param->tensor_shape.size; j++)
        {
            param->data[j] += lr * param->grad[j];
        }
    }
}

// Linear #################################################

// Linear #################################################

// Linear #################################################

// Linear #################################################
