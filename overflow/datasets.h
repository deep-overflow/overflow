#ifndef __DATASETS_H__
#define __DATASETS_H__

#include <iostream>
#include <cmath>
#include <random>
#include "core.h"

class Dataset;
class Sin;
class DataLoader;

class Dataset
{
public:
    Dataset();

    Tensor *input;
    Tensor *output;
    int n_samples;
};

class Sin : public Dataset
{
public:
    Sin(double start_ = 0, double end_ = 2 * M_PI, int n_samples_ = 1000);

    double start;
    double end;
};

class DataLoader
{
public:
    DataLoader(Dataset *dataset_, int batch_size_, bool shuffle_);

    Tensor batch(int idx);

    Dataset *dataset;
    int batch_size;
    bool shuffle;
};

#endif