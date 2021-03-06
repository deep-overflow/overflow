#include "core.h"

// Module #################################################

Module::Module() : func(NULL), params(NULL), n_func(0), n_params(0), verbose(false)
{
    if (verbose)
        std::cout << "Module::Module()" << std::endl;
}

Module::~Module()
{
    if (verbose)
        std::cout << "Module::~Module()" << std::endl;

    if (func != NULL)
    {
        for (int i = 0; i < n_func; i++)
            delete func[i];
        delete[] func;
    }

    if (params != NULL)
    {
        delete[] params;
    }
}

Tensor *Module::operator()(Tensor *input_)
{
    if (verbose)
        std::cout << "Tensor *Module::operator()(Tensor *input_)" << std::endl;

    Tensor *output = input_;

    for (int i = 0; i < n_func; i++)
    {
        output = (*func[i])(output);
    }

    return output;
}

void Module::add_params()
{
    if (verbose)
        std::cout << "void Module::add_params()" << std::endl;

    int idx = 0;

    params = new Tensor *[n_params];

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
    std::cout << "==================== module ====================" << std::endl;
    std::cout << n_func << " functions & " << n_params << " parameters" << std::endl;
    for (int i = 0; i < n_func; i++)
    {
        std::cout << i + 1 << "th function" << std::endl;
        func[i]->print();
    }
    std::cout << "==================== ****** ====================" << std::endl;
}

// Shape ##################################################

Shape::Shape() : dim(0), size(1), shape(NULL), verbose(false)
{
    if (verbose)
        std::cout << "Shape::Shape()" << std::endl;
}

Shape::Shape(const int *shape_, const int dim_) : dim(dim_), size(1), shape(NULL), verbose(false)
{
    if (verbose)
        std::cout << "Shape::Shape(const int *shape_, const int dim_)" << std::endl;

    if (dim > 0)
    {
        shape = new int[dim];
    }

    for (int i = 0; i < dim; i++)
    {
        shape[i] = shape_[i];
        size *= shape[i];
    }
}

Shape::Shape(const Shape &a) : dim(a.dim), size(1), shape(NULL), verbose(false)
{
    if (verbose)
        std::cout << "Shape::Shape(const Shape &a)" << std::endl;
    
    if (dim > 0)
    {
        shape = new int[dim];
    }

    for (int i = 0; i < dim; i++)
    {
        shape[i] = a.shape[i];
        size *= shape[i];
    }
}

Shape::~Shape()
{
    if (verbose)
        std::cout << "Shape::~Shape()" << std::endl;

    if (shape != NULL)
    {
        delete[] shape;
    }
}

void Shape::operator=(const Shape &a)
{
    /*
        No Memory Share.
    */
    if (verbose)
        std::cout << "void Shape::operator=(const Shape &a)" << std::endl;

    if (dim == a.dim)
    {
        size = 1;
    }
    else
    {
        dim = a.dim;
        size = 1;

        if (shape == NULL)
        {
            shape = new int[dim];
        }
        else
        {
            if (verbose)
                std::cout << "Re-Allocation in Shape" << std::endl;

            delete[] shape;
            shape = new int[dim];
        }
    }

    for (int i = 0; i < dim; i++)
    {
        shape[i] = a.shape[i];
        size *= shape[i];
    }
}

bool Shape::operator==(const Shape &a) const
{
    if (verbose)
        std::cout << "bool Shape::operator==(const Shape &a) const" << std::endl;

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

bool Shape::operator!=(const Shape &a) const
{
    if (verbose)
        std::cout << "bool Shape::operator!=(const Shape &a) const" << std::endl;
    
    if (dim != a.dim)
    {
        return true;
    }

    if (size != a.size)
    {
        return true;
    }

    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != a.shape[i])
        {
            return true;
        }
    }

    return false;
}

bool Shape::compare(const int *shape_, const int dim_)
{
    if (verbose)
        std::cout << "bool Shape::compare(const int *shape_, const int dim_)" << std::endl;
    
    if (dim != dim_)
    {
        return false;
    }

    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != shape_[i])
        {
            return false;
        }
    }

    return true;
}

bool Shape::compare(const Shape &a)
{
    if (verbose)
        std::cout << "bool Shape::compare(const Shape &a)" << std::endl;
    
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
    if (verbose)
        std::cout << "void Shape::reshape(const int *shape_, const int dim_)" << std::endl;
    
    if (compare(shape_, dim_))
    {
        if (verbose)
            std::cout << "Same Shape : Not need to reshape" << std::endl;
        
        return;
    }

    if (dim == dim_)
    {
        size = 1;
    }
    else
    {
        dim = dim_;
        size = 1;

        if (shape == NULL)
        {
            shape = new int[dim];
        }
        else
        {
            if (verbose)
                std::cout << "Re-Allocation in Shape" << std::endl;
            
            delete[] shape;
            shape = new int[dim];
        }
    }

    for (int i = 0; i < dim; i++)
    {
        shape[i] = shape_[i];
        size *= shape[i];
    }
}

void Shape::reshape(const Shape &a)
{
    if (verbose)
        std::cout << "void Shape::reshape(const Shape &a)" << std::endl;

    if (compare(a))
    {
        if (verbose)
            std::cout << "Same Shape : Not need to reshape" << std::endl;

        return;
    }

    if (dim == a.dim)
    {
        size = 1;
    }
    else
    {
        dim = a.dim;
        size = 1;

        if (shape == NULL)
        {
            shape = new int[dim];
        }
        else
        {
            if (verbose)
                std::cout << "Re-Allocation in Shape" << std::endl;

            delete[] shape;
            shape = new int[dim];
        }
    }

    for (int i = 0; i < dim; i++)
    {
        shape[i] = a.shape[i];
        size *= shape[i];
    }
}

void Shape::T()
{
    if (verbose)
        std::cout << "void Shape::T()" << std::endl;

    int *shape_;
    shape_ = new int[dim];

    for (int i = 0; i < dim; i++)
    {
        shape_[i] = shape[dim - 1 - i];
    }

    if (shape == NULL)
    {
        shape = shape_;
    }
    else
    {
        delete[] shape;
        shape = shape_;
    }
}

Shape Shape::index(int s, int e) const
{
    if (verbose)
        std::cout << "Shape Shape::index(int s, int e) const" << std::endl;

    if (e == -1)
    {
        e = dim;
    }

    if (s < 0 || dim <= s)
    {
        std::cerr << "Argument Error : 0 <= s < dim" << std::endl;
    }

    if (e < 0 || dim < e)
    {
        std::cerr << "Argument Error : 0 <= e <= dim" << std::endl;
    }

    if (s >= e)
    {
        std::cerr << "Argument Error : s < e" << std::endl;
    }

    // 0 <= s < e <= dim

    int dim_ = e - s;
    int *shape_ = new int[dim_];

    for (int i = s; i < e; i++)
    {
        shape_[i - s] = shape[i];
    }

    Shape c(shape_, dim_);

    return c;
}

void Shape::print() const
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

Tensor::Tensor() : func(NULL), requires_grad(true), verbose(false)
{
    if (verbose)
        std::cout << "Tensor::Tensor()" << std::endl;

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    std::random_device rd;
    std::mt19937 rng(rd());

    std::normal_distribution<double> normal(0, 1);

    data[0] = normal(rng);
    grad[0] = 1;
}

Tensor::Tensor(const double *data_, const int *shape_, const int dim_) : tensor_shape(shape_, dim_), func(NULL), requires_grad(true), verbose(false)
{
    if (verbose)
        std::cout << "Tensor::Tensor(const double *data_, const int *shape_, const int dim_)" << std::endl;

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_[i];
        grad[i] = 1;
    }
}

Tensor::Tensor(const double data_, const int *shape_, const int dim_) : tensor_shape(shape_, dim_), func(NULL), requires_grad(true), verbose(false)
{
    if (verbose)
        std::cout << "Tensor::Tensor(const double data_, const int *shape_, const int dim_)" << std::endl;

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_;
        grad[i] = 1;
    }
}

Tensor::Tensor(const int *shape_, const int dim_) : tensor_shape(shape_, dim_), func(NULL), requires_grad(true), verbose(false)
{
    if (verbose)
        std::cout << "Tensor::Tensor(const int *shape_, const int dim_)" << std::endl;

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    std::random_device rd;
    std::mt19937 rng(rd());

    std::normal_distribution<double> normal(0, 1);

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = normal(rng);
        grad[i] = 1;
    }
}

Tensor::Tensor(const double data_, const Shape &shape_) : tensor_shape(shape_), func(NULL), requires_grad(true), verbose(false)
{
    if (verbose)
        std::cout << "Tensor::Tensor(const double data_, const Shape &shape_)" << std::endl;

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_;
        grad[i] = 1;
    }
}

Tensor::Tensor(const Shape &shape_) : tensor_shape(shape_), func(NULL), requires_grad(true), verbose(false)
{
    if (verbose)
        std::cout << "Tensor::Tensor(const Shape &shape_)" << std::endl;

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

    std::random_device rd;
    std::mt19937 rng(rd());

    std::normal_distribution<double> normal(0, 1);

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = normal(rng);
        grad[i] = 1;
    }
}

Tensor::~Tensor()
{
    if (verbose)
        std::cout << "Tensor::~Tensor()" << std::endl;
    
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
    if (verbose)
        std::cout << "void Tensor::operator=(const Tensor &a)" << std::endl;

    if (tensor_shape != a.tensor_shape)
    {
        if (tensor_shape.size != a.tensor_shape.size)
        {
            tensor_shape = a.tensor_shape;
            
            if (verbose)
                std::cout << "Re-Allocation in Tensor" << std::endl;

            delete[] data;
            delete[] grad;

            data = new double[tensor_shape.size];
            grad = new double[tensor_shape.size];
        }
        else
        {
            tensor_shape = a.tensor_shape;
        }
    }

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
    if (verbose)
        std::cout << "Tensor Tensor::operator+(const Tensor &a)" << std::endl;

    if (tensor_shape != a.tensor_shape)
    {
        std::cerr << "Dimension Error in element-wise addition" << std::endl;
    }
    else
    {
        if (verbose)
            std::cout << "Dimension Match in element-wise addition" << std::endl;
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
    if (verbose)
        std::cout << "Tensor Tensor::operator-(const Tensor &a)" << std::endl;

    if (tensor_shape != a.tensor_shape)
    {
        std::cerr << "Dimension Error in element-wise substract" << std::endl;
    }
    else
    {
        if (verbose)
            std::cout << "Dimension Match in element-wise subtraction" << std::endl;
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
    if (verbose)
        std::cout << "Tensor Tensor::operator^(int k)" << std::endl;
    
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

Tensor Tensor::operator*(const Tensor &a)
{
    if (verbose)
        std::cout << "Tensor Tensor::operator*(const Tensor &a)" << std::endl;

    if (tensor_shape != a.tensor_shape)
    {
        std::cerr << "Dimension Error in element-wise multiplication" << std::endl;
    }
    else
    {
        if (verbose)
            std::cout << "Dimension Match in element-wise multiplication" << std::endl;
    }

    Tensor c(0.0, tensor_shape);

    for (int i = 0; i < tensor_shape.size; i++)
    {
        c.data[i] = data[i] * a.data[i];
    }

    return c;
}

Tensor Tensor::operator*(double k)
{
    if (verbose)
        std::cout << "Tensor Tensor::operator*(double k)" << std::endl;

    Tensor c(0.0, tensor_shape);

    for (int i = 0; i < tensor_shape.size; i++)
    {
        c.data[i] = k * data[i];
    }

    return c;
}

void Tensor::init(const double data_, const int *shape_, const int dim_)
{
    if (verbose)
        std::cout << "void Tensor::init(const double data_, const int *shape_, const int dim_)" << std::endl;

    tensor_shape.reshape(shape_, dim_);

    if (verbose)
        std::cout << "Re-Allocation in Tensor" << std::endl;

    delete[] data;
    delete[] grad;

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

void Tensor::init_like(const double data_, const Shape& shape_)
{
    if (verbose)
        std::cout << "void Tensor::init_like(const double data_, const Shape& shape_)" << std::endl;

    if (tensor_shape != shape_)
    {
        tensor_shape = shape_;

        if (verbose)
            std::cout << "Re-Allocation in Tensor" << std::endl;

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

    func = NULL;
    requires_grad = true;
}

void Tensor::init_like(const double data_, const Tensor &a)
{
    if (verbose)
        std::cout << "void Tensor::init_like(const double data_, const Tensor &a)" << std::endl;

    init_like(data_, a.tensor_shape);
}

void Tensor::random(const int *shape_, const int dim_, char init_)
{
    if (verbose)
        std::cout << "void Tensor::random(const int *shape_, const int dim_, char init_)" << std::endl;

    tensor_shape.reshape(shape_, dim_);

    if (verbose)
        std::cout << "Re-Allocation in Tensor" << std::endl;

    delete[] data;
    delete[] grad;

    data = new double[tensor_shape.size];
    grad = new double[tensor_shape.size];

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
    else if (init_ == 'h')
    {        
        double n_in = (double)tensor_shape.shape[0];
        double var = sqrt(2) / sqrt(n_in);
        std::normal_distribution<double> normal(0, var);

        for (int i = 0; i < tensor_shape.size; i++)
        {
            data[i] = normal(rng);
            grad[i] = 1;
        }
    }

    func = NULL;
    requires_grad = true;
}

void Tensor::random(const Shape &shape_, char init_)
{
    if (verbose)
        std::cout << "void Tensor::random(const Shape &shape_, char init_)" << std::endl;

    if (tensor_shape != shape_)
    {
        tensor_shape = shape_;

        if (verbose)
            std::cout << "Re-Allocation in Tensor" << std::endl;
        
        delete[] data;
        delete[] grad;

        data = new double[tensor_shape.size];
        grad = new double[tensor_shape.size];
    }

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
    else if (init_ == 'h')
    {
        double n_in = (double)tensor_shape.shape[0];
        double var = sqrt(2) / sqrt(n_in);
        std::normal_distribution<double> normal(0, var);

        for (int i = 0; i < tensor_shape.size; i++)
        {
            data[i] = normal(rng);
            grad[i] = 1;
        }
    }

    func = NULL;
    requires_grad = true;
}

void Tensor::random(char init_)
{
    // init_ == 'n' : normal distribution
    // init_ == 'u' : uniform distribution
    if (verbose)
        std::cout << "void Tensor::random(char init_)" << std::endl;

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

    func = NULL;
    requires_grad = true;
}

void Tensor::arange()
{
    if (verbose)
        std::cout << "void Tensor::arange()" << std::endl;

    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = i + 1;
        grad[i] = 1;
    }
}

Tensor Tensor::argmax() const
{
    int batch_size = tensor_shape.shape[0];
    int n_features = tensor_shape.size / batch_size;

    int shape_[] = {batch_size, 1};
    Tensor c(0.0, shape_, 2);

    for (int batch = 0; batch < batch_size; batch++)
    {
        double score = 0.0;
        
        for (int i = 0; i < n_features; i++)
        {
            int idx = batch * n_features + i;

            if (score < data[idx])
            {
                score = data[idx];

                c.data[batch] = i;
            }
        }
    }

    return c;
}

Tensor Tensor::index(int arg_num, ...) const
{
    if (verbose)
        std::cout << "Tensor Tensor::index(int arg_num, ...) const" << std::endl;

    va_list list;
    va_start(list, arg_num);

    int *arg = new int[arg_num];

    for (int i = 0; i < arg_num; i++)
    {
        arg[i] = va_arg(list, int); // ?????????
    }

    int size = 1; // result.data??? ??????
    int dim_ = tensor_shape.dim - arg_num;
    int *shape_ = new int[dim_]; // result??? shape

    for (int i = arg_num; i < tensor_shape.dim; i++)
    {
        shape_[i - arg_num] = tensor_shape.shape[i];
        size *= tensor_shape.shape[i];
    }

    Tensor result(1.0, shape_, dim_);

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

double Tensor::index_(int arg, ...) const
{
    if (verbose)
        std::cout << "double Tensor::index_(int arg, ...) const" << std::endl;

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
    if (verbose)
        std::cout << "double Tensor::grad_index(int arg, ...) const" << std::endl;

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

double Tensor::sum_()
{
    double value = 0;
    for (int i = 0; i < tensor_shape.size; i++)
    {
        value += data[i];
    }

    return value;
}
/*
Tensor Tensor::sum(int axis)
{
    Tensor c;

    if (axis == -1)
    {
        c.data[0] = 
    }
    else
    {

    }
}
*/
void Tensor::append(const Tensor &a, bool new_axis)
{
    /*
        new_axis??? true??????, batch??? ????????????.
    */
    if (verbose)
        std::cout << "void Tensor::append(const Tensor &a, bool new_axis)" << std::endl;

    if (tensor_shape.dim > 1)
    {
        if (a.tensor_shape != tensor_shape.index(1, -1))
        {
            std::cerr << "Dimension Error : Unmatching Dimension for append" << std::endl;
        }
    }

    int size = tensor_shape.size + a.tensor_shape.size;

    if (verbose)
        std::cout << "Re-Allocation in Tensor" << std::endl;

    double *data_ = new double[size];
    double *grad_ = new double[size];

    for (int i = 0; i < tensor_shape.size; i++)
    {
        if (tensor_shape.dim == 0)
        {
            data_[0] = a.data[0];
            grad_[0] = 1;
            break;
        }

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
        if (new_axis)
        {
            int dim_ = a.tensor_shape.dim + 1;
            int *shape_ = new int[dim_];

            shape_[0] = 1;
            for (int i = 1; i < dim_; i++)
            {
                shape_[i] = a.tensor_shape.shape[i - 1];
            }

            tensor_shape.reshape(shape_, dim_);
        }
        else{
            tensor_shape = a.tensor_shape;
        }
    }
    else
    {
        if (new_axis)
        {
            int dim_ = tensor_shape.dim;
            int *shape_ = new int[dim_];

            shape_[0] = tensor_shape.shape[0] + 1;
            for (int i = 1; i < dim_; i++)
            {
                shape_[i] = tensor_shape.shape[i];
            }

            tensor_shape.reshape(shape_, dim_);
        }
        else
        {
            int dim_ = tensor_shape.dim;
            int *shape_ = new int[dim_];

            shape_[0] = tensor_shape.shape[0] + a.tensor_shape.shape[0];
            for (int i = 1; i < dim_; i++)
            {
                shape_[i] = tensor_shape.shape[i - 1];
            }

            tensor_shape.reshape(shape_, dim_);
        }
    }

    delete[] data;
    delete[] grad;

    data = data_;
    grad = grad_;
}

void Tensor::backward()
{
    if (verbose)
        std::cout << "void Tensor::backward()" << std::endl;

    if (func != NULL)
    {
        func->backward();
    }
}

void Tensor::zero_grad()
{
    if (verbose)
        std::cout << "void Tensor::zero_grad()" << std::endl;

    for (int i = 0; i < tensor_shape.size; i++)
    {
        grad[i] = 1;
    }
    
    if (func != NULL)
    {
        func->zero_grad();
        func = NULL;
    }
}

void Tensor::dot(const Tensor &a)
{ // not generalized: for matrix
    if (verbose)
        std::cout << "void Tensor::dot(const Tensor &a)" << std::endl;

    if (tensor_shape.shape[1] != a.tensor_shape.shape[0])
    {
        std::cerr << "Dimension Error in dot function" << std::endl;
    }

    // (m x n) * (n x k) = m x k
    int m = tensor_shape.shape[0];
    int n = tensor_shape.shape[1]; // = a.tensor_shape.shape[0]
    int k = a.tensor_shape.shape[1];

    int shape_[] = {m, k};

    if (verbose)
        std::cout << "Re-Allocation in Tensor" << std::endl;

    double *data_ = new double[m * k];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            double value = 0;
            for (int t = 0; t < n; t++)
            {
                value += index_(i, t) * a.index_(t, j);
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
    if (verbose)
        std::cout << "void Tensor::grad_dot(const Tensor &a)" << std::endl;

    if (tensor_shape.shape[1] != a.tensor_shape.shape[0])
    {
        std::cerr << "Dimension Error in dot function" << std::endl;
    }

    // (m x n) * (n x k) = m x k
    int m = tensor_shape.shape[0];
    int n = tensor_shape.shape[1]; // =a.tensor_shape.shape[0]
    int k = a.tensor_shape.shape[1];

    int shape_[] = {m, k};

    if (verbose)
        std::cout << "Re-Allocation in Tensor" << std::endl;
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
            int idx_ = i * k + j;
            grad_[idx_] = value;
        }
    }

    tensor_shape.reshape(shape_, 2);

    delete[] grad;
    grad = grad_;
}

void Tensor::reshape(const int *shape_, const int dim_)
{
    if (verbose)
        std::cout << "void Tensor::reshape(const int *shape_, const int dim_)" << std::endl;
    
    int size = 1;

    for (int i = 0; i < dim_; i++)
    {
        size *= shape_[i];
    }

    if (size != tensor_shape.size)
    {
        std::cerr << "Reshape Unavailable" << std::endl;
    }
    else
    {
        tensor_shape.reshape(shape_, dim_);
    }
}

void Tensor::reshape(const Shape &a)
{
    if (verbose)
        std::cout << "void Tensor::reshape(const Shape &a)" << std::endl;
    
    if (a.size != tensor_shape.size)
    {
        std::cerr << "Reshape Unavailable" << std::endl;
    }
    else
    {
        tensor_shape.reshape(a);
    }
}

void Tensor::T()
{ // not generalized: for matrix
    if (verbose)
        std::cout << "void Tensor::T()" << std::endl;

    if (verbose)
        std::cout << "Re-Allocation in Tensor" << std::endl;

    double *data_;
    double *grad_;

    data_ = new double[tensor_shape.size];
    grad_ = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.shape[0]; i++)
    {
        for (int j = 0; j < tensor_shape.shape[1]; j++)
        {
            int idx_ = j * tensor_shape.shape[0] + i;
            data_[idx_] = index_(i, j);
            grad_[idx_] = grad_index(i, j);
        }
    }

    tensor_shape.T();

    delete[] data;
    delete[] grad;
    data = data_;
    grad = grad_;
}

void Tensor::print() const
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
    std::cout << std::endl;

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
    std::cout << std::endl;
}

Tensor dot(const Tensor &a, const Tensor &b, bool verbose)
{ // not generalized: for matrix
    if (verbose)
        std::cout << "Tensor dot(const Tensor &a, const Tensor &b)" << std::endl;

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
                double a_ = a.index_(i, t);
                double b_ = b.index_(t, j);
                value += a_ * b_;
            }
            int index_ = i * c.tensor_shape.shape[1] + j;
            c.data[index_] = value;
        }
    }

    return c;
}

// Function ###############################################

Function::Function() : input(NULL), input2(NULL), output(NULL), has_params(false), name("function"), verbose(false)
{
    if (verbose)
        std::cout << "Function::Function()" << std::endl;
}

Tensor *Function::operator()(Tensor *input_)
{
    if (verbose)
        std::cout << "Tensor *Function::operator()(Tensor *input_)" << std::endl;

    std::cerr << "Not Implemented Error" << std::endl;
    return NULL;
}

Tensor *Function::operator()(Tensor *output_, Tensor *label_)
{
    if (verbose)
        std::cout << "Tensor *Function::operator()(Tensor *output_, Tensor *label_)" << std::endl;

    std::cerr << "Not Implemented Error" << std::endl;
    return NULL;
}

void Function::backward()
{
    if (verbose)
        std::cout << "void Function::backward()" << std::endl;

    std::cerr << "Not Implemented Error" << std::endl;
}

void Function::zero_grad()
{
    if (verbose)
        std::cout << "void Function::zero_grad()" << std::endl;

    std::cerr << "Not Implemented Error" << std::endl;
}

Tensor *Function::return_params()
{
    if (verbose)
        std::cout << "Tensor *Function::return_params()" << std::endl;

    std::cerr << "Not Implemented Error" << std::endl;
    return NULL;
}

void Function::print()
{
    if (verbose)
        std::cout << "void Function::print()" << std::endl;
    
    std::cerr << "Not Implemented Error" << std::endl;
}
