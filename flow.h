#ifndef __FLOW_H__
#define __FLOW_H__

#include <iostream>
#include <string>

class Shape;
template <class T>
class Tensor;

class Shape
{
public:
    Shape();

};

template <class T>
class Tensor
{
public:
    Tensor();

    Tensor *tensor;
    T *data;
};

#endif