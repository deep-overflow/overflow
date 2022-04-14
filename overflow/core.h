#include <iostream>

class Shape;
class Tensor;
class Function;

class Shape
{
public:
    Shape();
    Shape(const int *shape_, const int dim_);

    bool operator==(const Shape &a);

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

    void operator=(const Tensor& a);
    Tensor operator+(const Tensor& a);
    // void operator*(const Tensor& c);

    double index(int i, int j); // for matrix

    void dot(Tensor a);
    void T();

    void print();

    // variable
    double *data;
    Shape tensor_shape;
};

Tensor dot(Tensor a, Tensor b);

class Function
{
public:
};
