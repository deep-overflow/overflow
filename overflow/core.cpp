#include "core.h"

// Shape ##################################################

Shape::Shape() {
    shape = NULL;
    dim = 0;
    size = 0;
}

Shape::Shape(int *shape_, int dim_) {
    dim = dim_;
    size = 1;
    shape = new int(dim);
    for (int i = 0; i < dim; i++) {
        shape[i] = shape_[i];
        size *= shape[i];
    }
}

void Shape::T() {
    int *shape_ = new int(dim);

    for (int i = 0; i < dim; i++) {
        shape_[i] = shape[i];
    }

    for (int i = 0; i < dim; i++) {
        shape[i] = shape_[dim - 1 - i];
    }

    delete shape_;
}

void Shape::print() {
    std::cout << "dim : " << dim << std::endl;
    std::cout << "size : " << size << std::endl;
    std::cout << "shape : [ ";
    for (int i = 0; i < dim; i++) {
        std::cout << shape[i] << " ";
    }
    std::cout << "]" << std::endl;
}

// Tensor #################################################

Tensor::Tensor() {
    tensor_shape = new Shape;
    data = NULL;
}

Tensor::Tensor(float *data_, int *shape_, int dim_) {
    tensor_shape = new Shape(shape_, dim_);

    data = new float(tensor_shape->size);
    for (int i = 0; i < tensor_shape->size; i++) {
        data[i] = data_[i];
    }
}

Tensor::Tensor(float data_, int *shape_, int dim_) {
    tensor_shape = new Shape(shape_, dim_);

    data = new float(tensor_shape->size);
    for (int i = 0; i < tensor_shape->size; i++) {
        data[i] = data_;
    }
}

void Tensor::T() { // not generalized

    tensor_shape->T();
}

void Tensor::print() {
    tensor_shape->print();

    std::cout << "data : " << std::endl;
    for (int i = 0; i < tensor_shape->size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################
