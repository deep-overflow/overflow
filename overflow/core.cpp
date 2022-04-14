#include "core.h"

// Shape ##################################################

Shape::Shape()
{
    std::cout << "Shape::Shape()" << std::endl;
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
}

Tensor::Tensor(const double *data_, const int *shape_, const int dim_)
{
    tensor_shape.init(shape_, dim_);

    data = new double[tensor_shape.size];
    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_[i];
    }
}

Tensor::Tensor(const double data_, const int *shape_, const int dim_)
{
    tensor_shape.init(shape_, dim_);

    data = new double[tensor_shape.size];
    for (int i = 0; i < tensor_shape.size; i++)
    {
        data[i] = data_;
    }
}

void Tensor::operator=(const Tensor &a)
{
    if (tensor_shape.size == a.tensor_shape.size) {
        tensor_shape.init(a.tensor_shape.shape, a.tensor_shape.dim);
    }
    else {
        tensor_shape.init(a.tensor_shape.shape, a.tensor_shape.dim);
        
        if (data == NULL) {
            data = new double[tensor_shape.size];
        }
        else {
            delete[] data;
            data = new double[tensor_shape.size];
        }
    }

    for (int i = 0; i < tensor_shape.size; i++) {
        data[i] = a.data[i];
    }
}

double Tensor::index(int i, int j)
{
    int index_ = tensor_shape.shape[1] * i + j;
    return data[index_];
}

void Tensor::dot(const Tensor &a)
{ // not generalized: for matrix

}

void Tensor::T()
{ // not generalized: for matrix

    double *data_;
    data_ = new double[tensor_shape.size];

    for (int i = 0; i < tensor_shape.shape[0]; i++)
    {
        for (int j = 0; j < tensor_shape.shape[1]; j++)
        {
            int index_ = j * tensor_shape.shape[0] + i;
            data_[index_] = index(i, j);
        }
    }

    tensor_shape.T();

    delete[] data;
    data = data_;
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
}

// Tensor dot(const Tensor a, const Tensor b) {
//     return Tensor();
// }

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################
