#include "nn.h"

// Linear #################################################

Linear::Linear(int in_features_, int out_features_)
{
    int shape_[] = {out_features_, in_features_};

    params.init(1.0, shape_, 2);
}

Tensor *Linear::operator()(Tensor *input_)
{
    if (output == NULL)
    {
        output = new Tensor;
    }

    *output = dot(params, *input_);
    output->func = this;

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

    std::cout << "Linear::backward()" << std::endl;

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
                    value += output->grad_index(i, t) * input->index(t, j);
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
                value += params.index(i, t) * output->grad_index(t, j);
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

void Linear::print()
{
    std::cout << "===== Linear class =====" << std::endl;
    params.print();
}

// ReLU ###################################################

ReLU::ReLU()
{
    std::cout << "ReLU::ReLU()" << std::endl;
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
    std::cout << "ReLU::backward()" << std::endl;

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

void ReLU::print()
{
    std::cout << "===== ReLU class =====" << std::endl;
}

// MSELoss ################################################

MSELoss::MSELoss()
{
    std::cout << "MSELoss::MSELoss()" << std::endl;
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

void MSELoss::print()
{
    std::cout << "===== MSELoss class =====" << std::endl;
}

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################
