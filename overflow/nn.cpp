#include "nn.h"

// Linear #################################################

Linear::Linear(int in_features_, int out_features_, char init_)
{
    /*
        O [batch, out_features] = I [batch, in_features] * P [in_features, out_features]
    */
    std::cout << "Linear::Linear(int in_features_, int out_features_, char init_)" << std::endl;

    init = init_;

    int shape_[] = {in_features_, out_features_};

    if (init)
    {
        params.random(shape_, 2, init);
    }
    else
    {
        params.init(1.0, shape_, 2);
    }

    has_params = true;

    name = "< Linear class : Function class >";
}

Tensor *Linear::operator()(Tensor *input_)
{
    std::cout << "Tensor *Linear::operator()(Tensor *input_)" << std::endl;
    
    if (output == NULL)
    {
        std::cout << "new Tensor" << std::endl;
        output = new Tensor();
        std::cout << "new Tensor" << std::endl;
    }

    // O [batch, out_feaures] = I [batch, in_features] * P [in_features, out_features]
    *output = dot(*input_, params);
    output->func = this;
    output->print();
    
    input = input_;

    return output;
}

void Linear::backward()
{ // not generalized: for matrix
    /*
    P : params, n x m
    I : input, m x k
    O : output, n x k
    O = PI : (n x m) * (m x k) -> n x k

    P.grad = O.grad I^T : (n x k) * (k x m) -> n x m
    I.grad = P^T O.grad : (m x n) * (n x k) -> m x k
    */
    int n = params.tensor_shape.shape[0];
    int m = params.tensor_shape.shape[1];
    int m_ = input->tensor_shape.shape[0];
    int k = input->tensor_shape.shape[1];

    // compute params.grad
    if (params.requires_grad)
    {
        input->T();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                double value = 0;
                for (int t = 0; t < k; t++)
                {
                    // (n x k) * (k x m) -> n x m
                    value += output->grad_index(i, t) * input->index_(t, j);
                }
                int index_ = i * m + j;
                params.grad[index_] = value;
            }
        }
        input->T();
    }

    // compute input->grad
    params.T();
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double value = 0;
            for (int t = 0; t < n; t++)
            {
                // (m x n) * (n x k) -> m x k
                value += params.index_(i, t) * output->grad_index(t, j);
            }
            int index_ = i * k + j;
            input->grad[index_] = value;
        }
    }
    params.T();

    // execute input->backward()
    if (input->func != NULL)
    {
        input->backward();
    }
}

void Linear::zero_grad()
{
    delete output;

    params.zero_grad();
    
    input->zero_grad();

    output = NULL;
    input = NULL;
}

Tensor *Linear::return_params()
{
    return &params;
}

void Linear::print()
{
    std::cout << name << std::endl;

    if (init == 0)
    {
        std::cout << "params init : NULL" << std::endl
                  << std::endl;
    }
    else if (init == 'n')
    {
        std::cout << "params init : normal" << std::endl
                  << std::endl;
    }
    else if (init == 'u')
    {
        std::cout << "params init : uniform" << std::endl
                  << std::endl;
    }
    else
    {
        std::cout << "params init : Unknown" << std::endl
                  << std::endl;
    }

    if (input == NULL)
    {
        std::cout << "input : NULL" << std::endl;
    }
    else
    {
        input->print();
    }

    if (input2 == NULL)
    {
        std::cout << "input2 : NULL" << std::endl;
    }
    else
    {
        input2->print();
    }

    if (output == NULL)
    {
        std::cout << "output : NULL" << std::endl;
    }
    else
    {
        output->print();
    }

    params.print();
}

// ReLU ###################################################

ReLU::ReLU()
{
    has_params = false;

    input = NULL;
    input2 = NULL;
    output = NULL;

    name = "< ReLU class : Function class >";
}

Tensor *ReLU::operator()(Tensor *input_)
{
    if (output == NULL)
    {
        output = new Tensor;
    }
    
    *output = *input_;
    for (int i = 0; i < output->tensor_shape.size; i++)
    {
        if (output->data[i] < 0)
        {
            output->data[i] = 0;
        }
    }
    output->func = this;
    
    input = input_;

    return output;
}

void ReLU::backward()
{
    for (int i = 0; i < output->tensor_shape.size; i++)
    {
        if (output->data[i] > 0)
        {
            input->grad[i] = output->grad[i];
        }
        else
        {
            input->grad[i] = 0;
        }
    }

    if (input->func != NULL)
    {
        input->backward();
    }
}

void ReLU::zero_grad()
{
    delete output;

    input->zero_grad();

    output = NULL;
    input = NULL;
}

void ReLU::print()
{
    std::cout << name << std::endl;
}

// MSELoss ################################################

MSELoss::MSELoss()
{
    has_params = false;

    input = NULL;
    input2 = NULL;
    output = NULL;

    name = "< MSELoss class : Function class >";
}

Tensor *MSELoss::operator()(Tensor *input_1, Tensor *input_2)
{
    if (output == NULL)
    {
        output = new Tensor;
    }

    *output = (*input_1 - *input_2) ^ 2;
    output->func = this;

    input = input_1;
    input2 = input_2;

    return output;
}

void MSELoss::backward()
{
    for (int i = 0; i < output->tensor_shape.size; i++)
    {
        double grad_input = 2 * (input->data[i] - input2->data[i]);

        input->grad[i] = output->grad[i];
        input->grad[i] *= grad_input;

        input2->grad[i] = output->grad[i];
        input2->grad[i] *= -grad_input;
    }

    if (input->func != NULL)
    {
        input->backward();
    }
    if (input2->func != NULL)
    {
        input2->backward();
    }
}

void MSELoss::zero_grad()
{
    delete output;

    input->zero_grad();
    input2->zero_grad();

    output = NULL;
    input = NULL;
    input2 = NULL;
}

void MSELoss::print()
{
    std::cout << name << std::endl;
}
