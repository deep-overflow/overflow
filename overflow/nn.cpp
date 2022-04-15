#include "nn.h"

// Linear #################################################

Linear::Linear(int in_features_, int out_features_)
{
    int shape_[] = {out_features_, in_features_};

    params.init(1.0, shape_, 2);
}

Tensor *Linear::operator()(Tensor *input_)
{
    if (output == NULL) {
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
    int n = params.tensor_shape.shape[0];
    int m = params.tensor_shape.shape[1];
    int m_ = input->tensor_shape.shape[0];
    int k = input->tensor_shape.shape[1];

    // compute params.grad
    if (params.requires_grad) {
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
            for (int t = 0; t < n;  t++)
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

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################
