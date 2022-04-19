#include "datasets.h"

// Dataset ################################################

Dataset::Dataset()
{
    input = NULL;
    output = NULL;
    n_samples = 0;
}

// Sin ####################################################

Sin::Sin(double start_, double end_, int n_samples_)
{
    start = start_;
    end = end_;
    n_samples = n_samples_;

    int shape_[] = {n_samples_, 1};
    input = new Tensor(0.0, shape_, 2);
    output = new Tensor(0.0, shape_, 2);

    double div = (double)(n_samples - 1);

    for (int i = 0; i < n_samples; i++)
    {
        double k = (double)i;
        input->data[i] = start + (end - start) * k / div;
        output->data[i] = sin(input->data[i]);
    }
}

// DataLoader #############################################

DataLoader::DataLoader(Dataset *dataset_, int batch_size_, bool shuffle_)
{
    dataset = dataset_;
    batch_size = batch_size_;
    shuffle = shuffle_;

    if (shuffle)
    {
    }
}

Tensor DataLoader::batch(int idx)
{
    Tensor *batch = new Tensor[batch_size];
}

// Optimizer ##############################################

// Optimizer ##############################################
