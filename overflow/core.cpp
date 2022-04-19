#include "core.h"

// Shape ##################################################

Shape::Shape()
{
    shape = NULL;
    dim = 0;
    size = 0;
}

Shape::Shape(const int *shape_, const int dim_)
{
    dim = dim_;
    size = 1;
    shape = new int[dim];
    for (int i = 0; i < dim; i++)
    {
        shape[i] = shape_[i];
        size *= shape[i];
    }
}

void Shape::operator=(const Shape &a)
{
    if (dim == a.dim)
    {
        size = 1;
    }
    else
    {
        dim = a.dim;
        size = 1;
        if (shape != NULL)
        {
            delete[] shape;
        }
        shape = new int[dim];
    }

    for (int i = 0; i < dim; i++)
    {
        shape[i] = a.shape[i];
        size *= shape[i];
    }
}

bool Shape::operator==(const Shape &a)
{
    if (dim != a.dim)
    {
        return false;
    }

    if (size != a.size)
    {
        return false;
    }

    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != a.shape[i])
        {
            return false;
        }
    }

    return true;
}

void Shape::init(const int *shape_, const int dim_)
{
    if (shape == NULL)
    {
        dim = dim_;
        size = 1;
        shape = new int[dim_];
    }
    else if (dim != dim_)
    {
        dim = dim_;
        size = 1;
        delete[] shape;
        shape = new int[dim_];
    }
    else
    {
        size = 1;
    }

    for (int i = 0; i < dim; i++)
    {
        shape[i] = shape_[i];
        size *= shape[i];
    }
}

// void Tensor::random_init(const int *shape_, const int dim_, char init_)
// {
    
// }

void Tensor::random_init(char init_)
{
    // init_ == 'n' : normal distribution
    // init_ == 'u' : uniform distribution
    std::random_device rd;
    std::mt19937 rng(rd());

    if (init_ == 'n')
    {
        std::normal_distribution<double> normal(0, 1);
        
        for (int i = 0; i < tensor_shape.size; i++)
        {
            data[i] = normal(rng);
        }
    }
    else if (init_ == 'u')
    {
        std::uniform_real_distribution<double> uniform(-1, 1);

        for (int i = 0; i < tensor_shape.size; i++)
        {
            data[i] = uniform(rng);
        }
    }
}

void Shape::T()
{
    int *shape_;
    shape_ = new int[dim];

    for (int i = 0; i < dim; i++)
    {
        shape_[i] = shape[dim - 1 - i];
    }

    delete[] shape;
    shape = shape_;
}

void Shape::print()
{
    std::cout << "dim : " << dim << std::endl;
    std::cout << "size : " << size << std::endl;
    std::cout << "shape : [ ";
    for (int i = 0; i < dim; i++)
    {
        std::cout << shape[i] << " ";
    }
    std::cout << "]" << std::endl;
}

// Tensor #################################################

Tensor::Tensor()
{
    data = NULL;
    grad = NULL;
    func = NULL;
    requires_grad = true;
}

Tensor::Tensor(const double *data_, const int *shape_, const int dim_)
{
    tensor_shape.init(shape_, dim_);

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_[i];
        grad[i] = 1;
    }
    func = NULL;
    requires_grad = true;
}

Tensor::Tensor(const double data_, const int *shape_, const int dim_)
{
    tensor_shape.init(shape_, dim_);

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_;
        grad[i] = 1;
    }
    func = NULL;
    requires_grad = true;
}

Tensor::Tensor(const double data_, const Shape &shape_)
{
    tensor_shape = shape_;

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_;
        grad[i] = 1;
    }
    func = NULL;
    requires_grad = true;
}

Tensor::~Tensor()
{
    if (data != NULL)
    {
        delete[] data;
    }

    if (grad != NULL)
    {
        delete[] grad;
    }
}

void Tensor::operator=(const Tensor &a)
{
    if (!(tensor_shape == a.tensor_shape))
    {
        tensor_shape = a.tensor_shape;
    }

    if (data == NULL)
    {
        data = new double[tensor_shape.size];
        grad = new double[tensor_shape.size];
    }
    else
    {
        delete[] data;
        data = new double[tensor_shape.size];
        grad = new double[tensor_shape.size];
    }

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = a.data[i];
        grad[i] = 1;
    }
}

Tensor Tensor::operator+(const Tensor &a)
{
    if (!(tensor_shape == a.tensor_shape))
    {
        std::cerr << "Dimension Error in element-wise add" << std::endl;
    }

    Tensor c(0.0, tensor_shape);

    for (int i = 0; i < tensor_shape.size; i++)
    {
        c.data[i] = data[i] + a.data[i];
    }

    return c;
}

Tensor Tensor::operator-(const Tensor &a)
{
    if (!(tensor_shape == a.tensor_shape))
    {
        std::cerr << "Dimension Error in element-wise substract" << std::endl;
    }

    Tensor c(0.0, tensor_shape);

    for (int i = 0; i < tensor_shape.size; i++)
    {
        c.data[i] = data[i] - a.data[i];
    }

    return c;
}

Tensor Tensor::operator^(int k)
{
    Tensor c(0.0, tensor_shape);

    for (int i = 0; i < tensor_shape.size; i++)
    {
        double value = 1;
        for (int j = 0; j < k; j++)
        {
            value *= data[i];
        }
        c.data[i] = value;
    }

    return c;
}

void Tensor::init(const double data_, const int *shape_, const int dim_)
{
    tensor_shape.init(shape_, dim_);

    if (data == NULL)
    {
        data = new double[tensor_shape.size];
        grad = new double[tensor_shape.size];
    }
    else
    {
        delete[] data;
        delete[] grad;

        data = new double[tensor_shape.size];
        grad = new double[tensor_shape.size];
    }

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_;
        grad[i] = 1;
    }
}

double Tensor::sum(int axis)
{
    // axis에 대한 수정 필요.
    double value = 0;
    for (int i = 0; i < tensor_shape.size; i++)
    {
        value += data[i];
    }

    return value;
}

double Tensor::index(int i, int j) const
{ // not generalized: for matrix
    int index_ = tensor_shape.shape[1] * i + j;
    return data[index_];
}

double Tensor::grad_index(int i, int j) const
{ // not generalized: for matrix
    int index_ = tensor_shape.shape[1] * i + j;
    return grad[index_];
}

void Tensor::backward()
{
    if (func != NULL)
    {
        func->backward();
    }
}

void Tensor::zero_grad()
{
    if (func != NULL)
    {
        func->zero_grad();
    }
}

void Tensor::dot(const Tensor &a)
{ // not generalized: for matrix
    if (tensor_shape.shape[1] != a.tensor_shape.shape[0])
    {
        std::cerr << "Dimension Error in dot function" << std::endl;
    }

    // (m x n) * (n x k) = m x k
    int m = tensor_shape.shape[0];
    int n = tensor_shape.shape[1]; // =a.tensor_shape.shape[0]
    int k = a.tensor_shape.shape[1];

    int shape_[] = {m, k};

    double *data_ = new double[m * k];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double value = 0;
            for (int t = 0; t < n; t++)
            {
                value += index(i, t) * a.index(t, j);
            }
            int index_ = i * k + j;
            data_[index_] = value;
        }
    }

    tensor_shape.init(shape_, 2);

    delete[] data;
    data = data_;
}

void Tensor::grad_dot(const Tensor &a)
{ // not generalized: for matrix
    if (tensor_shape.shape[1] != a.tensor_shape.shape[0])
    {
        std::cerr << "Dimension Error in dot function" << std::endl;
    }

    // (m x n) * (n x k) = m x k
    int m = tensor_shape.shape[0];
    int n = tensor_shape.shape[1]; // =a.tensor_shape.shape[0]
    int k = a.tensor_shape.shape[1];

    int shape_[] = {m, k};

    double *grad_ = new double[m * k];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double value = 0;
            for (int t = 0; t < n; t++)
            {
                value += grad_index(i, t) * a.grad_index(t, j);
            }
            int index_ = i * k + j;
            grad_[index_] = value;
        }
    }

    tensor_shape.init(shape_, 2);

    delete[] grad;
    grad = grad_;
}

void Tensor::T()
{ // not generalized: for matrix

    double *data_;
    double *grad_;

    data_ = new double[tensor_shape.size];
    grad_ = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.shape[0]; i++)
    {
        for (int j = 0; j < tensor_shape.shape[1]; j++)
        {
            int index_ = j * tensor_shape.shape[0] + i;
            data_[index_] = index(i, j);
            grad_[index_] = grad_index(i, j);
        }
    }

    tensor_shape.T();

    delete[] data;
    delete[] grad;
    data = data_;
    grad = grad_;
}

void Tensor::print()
{ // not generalized: for matrix
    tensor_shape.print();

    std::cout << "data : " << std::endl;
    for (int i = 0; i < tensor_shape.shape[0]; i++)
    {
        for (int j = 0; j < tensor_shape.shape[1]; j++)
        {
            std::cout << index(i, j) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "grad : " << std::endl;
    for (int i = 0; i < tensor_shape.shape[0]; i++)
    {
        for (int j = 0; j < tensor_shape.shape[1]; j++)
        {
            std::cout << grad_index(i, j) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Tensor dot(const Tensor &a, const Tensor &b)
{ // not generalized: for matrix

    if (a.tensor_shape.shape[1] != b.tensor_shape.shape[0])
    {
        std::cerr << "Dimension Error in dot function" << std::endl;
    }

    // (m x n) * (n x k) = m x k
    int m = a.tensor_shape.shape[0];
    int n = a.tensor_shape.shape[1]; // = b.tensor_shape.shape[0]
    int k = b.tensor_shape.shape[1];

    int shape_[] = {m, k};

    Tensor c(0.0, shape_, 2);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double value = 0;
            for (int t = 0; t < n; t++)
            {
                double a_ = a.index(i, t);
                double b_ = b.index(t, j);
                value += a_ * b_;
            }
            int index_ = i * c.tensor_shape.shape[1] + j;
            c.data[index_] = value;
        }
    }

    return c;
}

// Function ###############################################

Function::Function()
{
    input = NULL;
    output = NULL;
}

Tensor *Function::operator()(Tensor *input_)
{
    std::cerr << "Not Implemented Error" << std::endl;
    return NULL;
}

Tensor *Function::operator()(Tensor *input_1, Tensor *input_2)
{
    std::cerr << "Not Implemented Error" << std::endl;
    return NULL;
}

void Function::backward()
{
    std::cerr << "Not Implemented Error" << std::endl;
}

void Function::zero_grad()
{
    std::cerr << "Not Implemented Error" << std::endl;
}

void Function::print()
{
    std::cerr << "Not Implemented Error" << std::endl;
}

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################
