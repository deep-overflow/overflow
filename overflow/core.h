#include <iostream>

class Shape;
class Tensor;
class Function;

class Shape
{
public:
    Shape();
    Shape(const int *shape_, const int dim_);

    void init(const int *shape_, const int dim_);

    void T();

    void print();

    // variable
    int *shape;
    int dim;
    int size;
};

class Tensor
{
public:
    Tensor();
    Tensor(const double *data_, const int *shape_, const int dim_);
    Tensor(const double data_, const int *shape_, const int dim_);

    void T();

    void print();

    // variable
    double *data;
    Shape tensor_shape;
};

class Function
{
public:
};
