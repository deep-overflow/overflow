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
}

Tensor DataLoader::operator()()
{
    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dis(0, dataset->n_samples - 1);

    int *batch_idx = new int[batch_size];

    for (int i = 0; i < batch_size; i++)
    {
        batch_idx[i] = dis(gen);
    }

    return batch(batch_idx);
}

Tensor DataLoader::batch(int *idx)
{
    // Tensor에 indexing 기능을 추가해야 한다.
    Shape shape = dataset->input->tensor_shape;
    shape.shape[0] = batch_size;

    Tensor batch;

    for (int i = 0; i < batch_size; i++)
    {
        int batch_idx = idx[i];

        // 구현해야 함.
        batch.append(dataset->input->index_(batch_idx));
    }

    return batch;
}
