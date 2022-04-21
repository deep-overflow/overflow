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
        output->data[i] = sin(input->data[i]) + 5.0;
    }
}

// DataLoader #############################################

DataLoader::DataLoader(Dataset *dataset_, int batch_size_, bool shuffle_)
{
    dataset = dataset_;
    batch_size = batch_size_;
    shuffle = shuffle_;
    batch_idx = new int[batch_size];
}

void DataLoader::batching()
{
    if (shuffle) {
        std::random_device rd;

        std::mt19937 gen(rd());

        std::uniform_int_distribution<int> dis(0, dataset->n_samples - 1);

        for (int i = 0; i < batch_size; i++)
        {
            batch_idx[i] = dis(gen);
        }
    }
}

Tensor DataLoader::input()
{
    std::cout << "input" << std::endl;
    Tensor batch_input;

    std::cout << "input" << std::endl;
    for (int i = 0; i < batch_size; i++)
    {
        int idx = batch_idx[i];
        Tensor batch = dataset->input->index_(1, idx);
        batch.print();
        batch_input.append(batch);
    }
    std::cout << "batch_input.print()" << std::endl;
    batch_input.print();
    std::cout << "input" << std::endl;
    return batch_input;
}

Tensor DataLoader::label()
{
    Tensor batch_label;

    for (int i = 0; i < batch_size; i++)
    {
        int idx = batch_idx[i];
        batch_label.append(dataset->output->index_(1, idx));
    }

    return batch_label;
}