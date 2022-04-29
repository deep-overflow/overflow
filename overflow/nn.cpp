#include "nn.h"

// Linear #################################################

Linear::Linear(int in_features_, int out_features_, char init_) : init(init_)
{
    /*
        O [batch, out_features] = I [batch, in_features] * P [in_features, out_features]
    */
    if (verbose)
        std::cout << "Linear::Linear(int in_features_, int out_features_, char init_)" << std::endl;

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

Linear::~Linear()
{
    if (verbose)
        std::cout << "Linear::~Linear" << std::endl;

    if (output != NULL)
    {
        delete output;
    }
}

Tensor *Linear::operator()(Tensor *input_)
{
    if (verbose)
        std::cout << "Tensor *Linear::operator()(Tensor *input_)" << std::endl;
    
    if (output == NULL)
    {
        output = new Tensor();
    }

    // O [batch, out_feaures] = I [batch, in_features] * P [in_features, out_features]
    *output = dot(*input_, params, verbose);
    output->func = this;
    
    input = input_;

    return output;
}

void Linear::backward()
{ // not generalized: for matrix
    /*
    I : inputs, n x m
    P : params, m x k
    O : outputs, n x k

    O = I P : (n x m) * (m x k) -> n x k

    P.grad = I^T O.grad : (m x n) * (n x k) -> m x k
    I.grad = O.grad P^T : (n x k) * (k x m) -> n x m
    */
    if (verbose)
        std::cout << "void Linear::backward()" << std::endl;

    int n = input->tensor_shape.shape[0];
    int m = input->tensor_shape.shape[1];
    int m_ = params.tensor_shape.shape[0];
    int k = params.tensor_shape.shape[1];
    
    // compute params.grad : (m x n) * (n x k) -> m x k
    if (params.requires_grad)
    {
        if (verbose)
            std::cout << "Compute params.grad" << std::endl;
        input->T();
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double value = 0;
                for (int t = 0; t < n; t++)
                {
                    // (n x k) * (k x m) -> n x m
                    value += input->index_(i, t) * output->grad_index(t, j);
                }
                int index_ = i * k + j;
                params.grad[index_] = value;
            }
        }
        input->T();
    }

    // compute input->grad : (n x k) * (k x m) -> n x m
    params.T();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double value = 0;
            for (int t = 0; t < k; t++)
            {
                // (m x n) * (n x k) -> m x k
                value += output->grad_index(i, t) * params.index_(t, j);
            }
            int index_ = i * m + j;
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
    if (verbose)
        std::cout << "void Linear::zero_grad()" << std::endl;

    delete output;

    params.zero_grad();
    
    input->zero_grad();

    output = NULL;
    input = NULL;
}

Tensor *Linear::return_params()
{
    if (verbose)
        std::cout << "Tensor *Linear::return_params()" << std::endl;

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
        std::cout << "input :" << std::endl;
        input->print();
    }

    if (input2 == NULL)
    {
        std::cout << "input2 : NULL" << std::endl;
    }
    else
    {
        std::cout << "input2 :" << std::endl;
        input2->print();
    }

    if (output == NULL)
    {
        std::cout << "output : NULL" << std::endl;
    }
    else
    {
        std::cout << "output :" << std::endl;
        output->print();
    }

    std::cout << "params :" << std::endl;
    params.print();
}

// DropOut ################################################

Dropout::Dropout(double ratio_) : ratio(ratio_), dropout(NULL), n_drop(0)
{
    if (verbose)
        std::cout << "DropOut::DropOut()" << std::endl;
    
    name = "< DropOut class : Function class >";
}

Dropout::~Dropout()
{
    if (verbose)
        std::cout << "Dropout::~Dropout()" << std::endl;
    
    if (output != NULL)
    {
        delete output;
    }

    if (dropout != NULL)
    {
        delete dropout;
    }
}

Tensor *Dropout::operator()(Tensor *input_)
{
    if (verbose)
        std::cout << "Tensor *Dropout::operator()(Tensor *input_)" << std::endl;
    
    if (output == NULL)
    {
        output = new Tensor;
    }

    *output = *input_;

    if (dropout == NULL)
    {
        dropout = new Tensor(1.0, input_->tensor_shape);
    }
    else
    {
        dropout->init_like(1.0, input_->tensor_shape);
    }

    int batch_size = dropout->tensor_shape.shape[0];
    int n_features = dropout->tensor_shape.size;
    n_features /= batch_size;
    n_drop = n_features * ratio;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, n_features - 1);

    for (int batch = 0; batch < batch_size; batch++)
    {
        for (int i = 0; i < n_drop; i++)
        {
            int idx = batch * n_features;
            idx += dis(gen);

            dropout->data[idx] = 0;
        }
    }

    *output = *output * *dropout;

    output->func = this;

    input = input_;

    return output;
}

void Dropout::backward()
{  
    if (verbose)
        std::cout << "void Dropout::backward()" << std::endl;
    
    for (int i = 0; i < output->tensor_shape.size; i++)
    {
        input->grad[i] = output->grad[i] * dropout->data[i];
    }

    if (input->func != NULL)
    {
        input->backward();
    }
}

void Dropout::zero_grad()
{
    if (verbose)
        std::cout << "void Dropout::zero_grad()" << std::endl;

    if (output != NULL)
    {
        delete output;
    }

    input->zero_grad();

    output = NULL;
    input = NULL;
}

void Dropout::print()
{
    std::cout << name << std::endl;
    std::cout << "ratio : " << ratio << std::endl;
    std::cout << "n_drop : " << n_drop << std::endl;

    if (input == NULL)
    {
        std::cout << "input : NULL" << std::endl;
    }
    else
    {
        std::cout << "input :" << std::endl;
        input->print();
    }

    if (input2 == NULL)
    {
        std::cout << "input2 : NULL" << std::endl;
    }
    else
    {
        std::cout << "input2 :" << std::endl;
        input2->print();
    }

    if (output == NULL)
    {
        std::cout << "output : NULL" << std::endl;
    }
    else
    {
        std::cout << "output :" << std::endl;
        output->print();
    }
}

// ReLU ###################################################

ReLU::ReLU()
{
    if (verbose)
        std::cout << "ReLU::ReLU()" << std::endl;

    name = "< ReLU class : Function class >";
}

ReLU::~ReLU()
{
    if (verbose)
        std::cout << "ReLU::~ReLU()" << std::endl;

    if (output != NULL)
    {
        delete output;
    }
}

Tensor *ReLU::operator()(Tensor *input_)
{
    if (verbose)
        std::cout << "Tensor *ReLU::operator()(Tensor *input_)" << std::endl;

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
    if (verbose)
        std::cout << "void ReLU::backward()" << std::endl;

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
    if (verbose)
        std::cout << "void ReLU::zero_grad()" << std::endl;

    if (output != NULL)
    {
        delete output;
    }

    input->zero_grad();

    output = NULL;
    input = NULL;
}

void ReLU::print()
{
    std::cout << name << std::endl;

    if (input == NULL)
    {
        std::cout << "input : NULL" << std::endl;
    }
    else
    {
        std::cout << "input :" << std::endl;
        input->print();
    }

    if (input2 == NULL)
    {
        std::cout << "input2 : NULL" << std::endl;
    }
    else
    {
        std::cout << "input2 :" << std::endl;
        input2->print();
    }

    if (output == NULL)
    {
        std::cout << "output : NULL" << std::endl;
    }
    else
    {
        std::cout << "output :" << std::endl;
        output->print();
    }
}

// Sigmoid ################################################

Sigmoid::Sigmoid()
{
    if (verbose)
        std::cout << "Sigmoid::Sigmoid()" << std::endl;
    
    name = "< Sigmoid class : Function class >";
}

Sigmoid::~Sigmoid()
{
    if (verbose)
        std::cout << "Sigmoid::~Sigmoid()" << std::endl;
    
    if (output != NULL)
    {
        delete output;
    }
}

Tensor *Sigmoid::operator()(Tensor *input_)
{
    if (verbose)
        std::cout << "Tensor *Sigmoid::operator()(Tensor *input_)" << std::endl;
    
    if (output != NULL)
    {
        delete output;
    }

    output = new Tensor(input_->tensor_shape);

    for (int i = 0; i < output->tensor_shape.size; i++)
    {
        double k = exp(input_->data[i]);
        output->data[i] = k / (1 + k);
    }

    output->func = this;

    input = input_;

    return output;
}

void Sigmoid::backward()
{
    if (verbose)
        std::cout << "void Sigmoid::backward()" << std::endl;
    
    for (int i = 0; i < output->tensor_shape.size; i++)
    {
        double k = output->data[i];
        k = k * (1 - k);
        input->grad[i] = k * output->grad[i];
    }

    if (input->func != NULL)
    {
        input->backward();
    }
}

void Sigmoid::zero_grad()
{
    if (verbose)
        std::cout << "void Sigmoid::zero_grad()" << std::endl;
    
    if (output != NULL)
    {
        delete output;
    }

    input->zero_grad();

    output = NULL;
    input = NULL;
}

void Sigmoid::print()
{
    std::cout << name << std::endl;

    if (input == NULL)
    {
        std::cout << "input : NULL" << std::endl;
    }
    else
    {
        std::cout << "input :" << std::endl;
        input->print();
    }

    if (input2 == NULL)
    {
        std::cout << "input2 : NULL" << std::endl;
    }
    else
    {
        std::cout << "input2 :" << std::endl;
        input2->print();
    }

    if (output == NULL)
    {
        std::cout << "output : NULL" << std::endl;
    }
    else
    {
        std::cout << "output :" << std::endl;
        output->print();
    }
}

// MSELoss ################################################

MSELoss::MSELoss()
{
    if (verbose)
        std::cout << "MSELoss::MSELoss()" << std::endl;
    
    name = "< MSELoss class : Function class >";
}

MSELoss::~MSELoss()
{
    if (verbose)
        std::cout << "MSELoss::~MSELoss()" << std::endl;

    if (output != NULL)
    {
        delete output;
    }
}

Tensor *MSELoss::operator()(Tensor *input_1, Tensor *input_2)
{
    if (verbose)
        std::cout << "Tensor *MSELoss::operator()(Tensor *input_1, Tensor *input_2)" << std::endl;

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
    if (verbose)
        std::cout << "void MSELoss::backward()" << std::endl;

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
    if (verbose)
        std::cout << "void MSELoss::zero_grad()" << std::endl;

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

    if (input == NULL)
    {
        std::cout << "input : NULL" << std::endl;
    }
    else
    {
        std::cout << "input :" << std::endl;
        input->print();
    }

    if (input2 == NULL)
    {
        std::cout << "input2 : NULL" << std::endl;
    }
    else
    {
        std::cout << "input2 :" << std::endl;
        input2->print();
    }

    if (output == NULL)
    {
        std::cout << "output : NULL" << std::endl;
    }
    else
    {
        std::cout << "output :" << std::endl;
        output->print();
    }
}
