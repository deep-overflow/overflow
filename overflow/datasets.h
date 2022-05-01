#ifndef __DATASETS_H__
#define __DATASETS_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include "core.h"

class Dataset;
class Sin;
class Circle;
class MNIST;
class DataLoader;

class Dataset
{
public:
    Dataset();

    Tensor *input;
    Tensor *output;
    int n_samples;
    bool verbose;
};

class Sin : public Dataset
{
public:
    Sin(double start_ = 0, double end_ = 2 * M_PI, int n_samples_ = 1000);

    double start;
    double end;
};

class Circle : public Dataset
{
public:
    Circle(double radius_, int n_x = 1000, int n_y = 1000);

    double radius;
};

class MNIST : public Dataset
{
public:
    MNIST(std::string path_, std::string type_ = "train");

    void visualize(int idx_);

    std::string type;
    std::string path;
};

class DataLoader
{
public:
    DataLoader(Dataset *dataset_, int batch_size_, bool shuffle_);

    void batching(); // shuffle에 따른 방식으로 batch를 생성한다.
    Tensor input(); // batching에서 생성된 input Tensor를 반환한다.
    Tensor label(); // batching에서 생성된 label Tensor를 반환한다.

    Dataset *dataset;
    int batch_size;
    int *batch_idx;
    bool shuffle;
    bool verbose;
};

#endif