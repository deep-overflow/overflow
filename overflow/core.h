#include <iostream>


class Shape;
class Tensor;
class Function;


class Shape
{
public:
    Shape();
    Shape(int *shape_, int dim_);

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
    Tensor(float *data_, int *shape_, int dim_);
    Tensor(float data_, int *shape_, int dim_);

    void T();

    void print();

    // variable
    Shape *tensor_shape;
    float *data;
};


class Function
{
public:

};
