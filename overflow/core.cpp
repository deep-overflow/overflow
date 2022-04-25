#include "core.h"

// Module #################################################

Module::Module()
{
    func = NULL;
    params = NULL;
    n_func = 0;
    n_params = 0;
}

Tensor *Module::operator()(Tensor *input_)
{
    Tensor *output = input_;

    for (int i = 0; i < n_func; i++)
    {
        output = (*func[i])(output);
    }

    return output;
}

void Module::add_params()
{
    int idx = 0;

    for (int i = 0; i < n_func; i++)
    {
        if (func[i]->has_params)
        {
            params[idx] = func[i]->return_params();
            idx++;
        }
    }
}

void Module::print()
{
    std::cout << "==================== funcs ====================" << std::endl;
    std::cout << n_func << " functions & " << n_params << " parameters" << std::endl;
    for (int i = 0; i < n_func; i++)
    {
        std::cout << i + 1 << "th function" << std::endl;
        func[i]->print();
    }
    std::cout << "==================== ***** ====================" << std::endl;
}

// Shape ##################################################

Shape::Shape()
{
    /*
        This Initializer is for 0 dimension Tensor,
        which means scalar.
    */
    dim = 0;
    size = 1;
    shape = new int[dim];
}

Shape::Shape(const int *shape_, const int dim_)
{
    /*
        This Initializer is for dim_ dimension Tensor.
        If dim_ = 1, Vector.
        If dim_ = 2, Matrix.
        If dim_ >= 3, Tensor.
    */
    dim = dim_;
    size = 1;
    shape = new int[dim];
    for (int i = 0; i < dim; i++)
    {
        shape[i] = shape_[i];
        size *= shape[i];
    }
}

Shape::Shape(const Shape& a)
{
    dim = a.dim;
    size = 1;
    shape = new int[dim];
    for (int i = 0; i < dim; i++)
    {
        shape[i] = a.shape[i];
        size *= shape[i];
    }
}

void Shape::operator=(const Shape &a)
{
    /*
        No Memory Share.
    */
    if (dim == a.dim)
    {
        size = 1;
    }
    else
    {
        dim = a.dim;
        size = 1;
        delete[] shape;
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

void Shape::reshape(const int *shape_, const int dim_)
{
    /*
        No Memory Share.
    */
    if (dim == dim_)
    {
        size = 1;
    }
    else
    {
        dim = dim_;
        size = 1;
        delete[] shape;
        shape = new int[dim];
    }

    for (int i = 0; i < dim; i++)
    {
        shape[i] = shape_[i];
        size *= shape[i];
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
    std::cout << "]" << std::endl
              << std::endl;
}

// Tensor #################################################

Tensor::Tensor()
{
    /*
        This Initializer is for 0 dimension Tensor,
        which means scalar.
    */
    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    data[0] = 1; // random init 하고 싶다.
    grad[0] = 1; // random init 하고 싶다.

    func = NULL;
    requires_grad = true;
}

Tensor::Tensor(const double *data_, const int *shape_, const int dim_)
{
    tensor_shape.reshape(shape_, dim_);

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
    tensor_shape.reshape(shape_, dim_);

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

    delete[] data;
    delete[] grad;
    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = a.data[i];
        grad[i] = 1;
    }

    func = NULL;
    requires_grad = true;
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
    tensor_shape.reshape(shape_, dim_);

    delete[] data;
    delete[] grad;
    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_;
        grad[i] = 1;
    }
}

void Tensor::random_init(const int *shape_, const int dim_, char init_)
{
    tensor_shape.reshape(shape_, dim_);

    delete[] data;
    delete[] grad;

    data = new double[tensor_shape.size];
    data = new double[tensor_shape.size];

    std::random_device rd;
    std::mt19937 rng(rd());

    if (init_ == 'n')
    {
        std::normal_distribution<double> normal(0, 1);

        for (int i = 0; i < tensor_shape.size; i++)
        {
            data[i] = normal(rng);
            grad[i] = 1;
        }
    }
    else if (init_ == 'u')
    {
        std::uniform_real_distribution<double> uniform(-1, 1);

        for (int i = 0; i < tensor_shape.size; i++)
        {
            data[i] = uniform(rng);
            grad[i] = 1;
        }
    }
}

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
            grad[i] = 1;
        }
    }
    else if (init_ == 'u')
    {
        std::uniform_real_distribution<double> uniform(-1, 1);

        for (int i = 0; i < tensor_shape.size; i++)
        {
            data[i] = uniform(rng);
            grad[i] = 1;
        }
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

void Tensor::append(const Tensor &a)
{
    int size = tensor_shape.size + a.tensor_shape.size;
    double *data_ = new double[size];
    double *grad_ = new double[size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data_[i] = data[i];
        grad_[i] = 1;
    }

    for (int i = tensor_shape.size; i < size; i++)
    {
        data_[i] = a.data[i - tensor_shape.size];
        grad_[i] = 1;
    }

    if (tensor_shape.dim == 0)
    {
        tensor_shape = a.tensor_shape;
    }
    else
    {
        tensor_shape.shape[0] = tensor_shape.shape[0] + a.tensor_shape.shape[0];
        tensor_shape.reshape(tensor_shape.shape, tensor_shape.dim);
    }

    if (data != NULL)
    {
        delete[] data;
        delete[] grad;
    }

    data = data_;
    grad = grad_;
}

Tensor Tensor::index_(int arg_num, ...) const
{
    va_list list;
    va_start(list, arg_num);

    int *arg = new int[arg_num];

    for (int i = 0; i < arg_num; i++)
    {
        arg[i] = va_arg(list, int); // 인덱싱
    }

    Tensor result;

    int size = 1; // result.data의 크기
    int dim_ = tensor_shape.dim - arg_num + 1;
    int *shape_ = new int[dim_]; // result의 shape

    shape_[0] = 1;
    for (int i = arg_num; i < tensor_shape.dim; i++)
    {
        shape_[i + 1 - arg_num] = tensor_shape.shape[i];
        size *= tensor_shape.shape[i];
    }

    result.init(1.0, shape_, dim_);

    int start = arg[0];
    
    for (int i = 1; i < tensor_shape.dim; i++)
    {
        start *= tensor_shape.shape[i];
        if (i < arg_num)
        {
            start += arg[i];
        }
    }

    for (int i = 0; i < size; i++)
    {
        result.data[i] = data[start + i];
    }

    va_end(list);

    delete[] arg;
    delete[] shape_;

    return result;
}

double Tensor::index(int arg, ...) const
{
    // indexing rule
    // - two type of indexing expression
    //     1. start:end:stride
    //     2. -1
    // - The maximum number of axis is 4,
    // - So there is 4 Axis class.
    
    va_list list;
    va_start(list, arg);

    int *idx = new int[tensor_shape.dim];

    idx[0] = arg;

    for (int i = 1; i < tensor_shape.dim; i++)
    {
        idx[i] = va_arg(list, int);
    }

    int index_ = idx[0];
    for (int i = 1; i < tensor_shape.dim; i++)
    {
        index_ *= tensor_shape.shape[i];
        index_ += idx[i];
    }

    va_end(list);

    delete[] idx;

    return data[index_];
}

double Tensor::grad_index(int arg, ...) const
{
    va_list list;
    va_start(list, arg);

    int *idx = new int[tensor_shape.dim];

    idx[0] = arg;

    for (int i = 1; i < tensor_shape.dim; i++)
    {
        idx[i] = va_arg(list, int);
    }

    int index_ = idx[0];
    for (int i = 1; i < tensor_shape.dim; i++)
    {
        index_ *= tensor_shape.shape[i];
        index_ += idx[i];
    }

    va_end(list);

    delete[] idx;

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
    for (int i = 0; i < tensor_shape.size; i++)
    {
        grad[i] = 1;
    }
    
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
    int n = tensor_shape.shape[1]; // = a.tensor_shape.shape[0]
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

    tensor_shape.reshape(shape_, 2);

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

    tensor_shape.reshape(shape_, 2);

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

    std::cout << "requires_grad : ";
    if (requires_grad)
    {
        std::cout << "true" << std::endl
                  << std::endl;
    }
    else
    {
        std::cout << "false" << std::endl
                  << std::endl;
    }

    std::cout << "func : ";
    if (func)
    {
        std::cout << func->name << std::endl
                  << std::endl;
    }
    else
    {
        std::cout << "NULL" << std::endl
                  << std::endl;
    }

    std::cout << "data : " << std::endl;
    for (int i = 0; i < tensor_shape.size; i++)
    {
        std::cout << data[i] << ' ';

        int k = 1;
        for (int j = 0; j < tensor_shape.dim; j++)
        {
            k *= tensor_shape.shape[tensor_shape.dim - 1 - j];

            if ((i + 1) % k == 0)
            {
                std::cout << std::endl;
            }
        }
    }

    std::cout << "grad : " << std::endl;
    for (int i = 0; i < tensor_shape.size; i++)
    {
        std::cout << grad[i] << ' ';

        int k = 1;
        for (int j = 0; j < tensor_shape.dim; j++)
        {
            k *= tensor_shape.shape[tensor_shape.dim - 1 - j];

            if ((i + 1) % k == 0)
            {
                std::cout << std::endl;
            }
        }
    }

    
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
    input2 = NULL;
    output = NULL;
    has_params = false;
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

Tensor *Function::return_params()
{
    std::cerr << "Not Implemented Error" << std::endl;

    return NULL;
}

void Function::print()
{
    std::cerr << "Not Implemented Error" << std::endl;
}
