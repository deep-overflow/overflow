#include "datasets.h"

// Dataset ################################################

Dataset::Dataset() : input(NULL), output(NULL), n_samples(0), verbose(false)
{
    verbose = true;
    if (verbose)
        std::cout << "Dataset::Dataset()" << std::endl;
}

// Sin ####################################################

Sin::Sin(double start_, double end_, int n_samples_) : start(start_), end(end_)
{
    if (verbose)
        std::cout << "Sin::sin(double start_, double end_, int n_samples_)" << std::endl;

    n_samples = n_samples_;

    int shape_[] = {n_samples_, 1};
    input = new Tensor(0.0, shape_, 2);
    output = new Tensor(0.0, shape_, 2);

    double div = (double)(n_samples - 1);

    for (int i = 0; i < n_samples; i++)
    {
        double k = (double)i;
        input->data[i] = start + (end - start) * k / div;
        if (sin(input->data[i]) > 0)
        {
            output->data[i] = 1.0;
        }
        else
        {
            output->data[i] = 0.0;
        }
    }
}

// Circle #################################################

Circle::Circle(double radius_, int n_x, int n_y) : radius(radius_)
{
    if (verbose)
        std::cout << "Circle::Circle(double radius_, int n_samples_)" << std::endl;
    
    n_samples = n_x * n_y;

    int shape_i[] = {n_samples, 2};
    int shape_o[] = {n_samples, 1};

    input = new Tensor(0.0, shape_i, 2);
    output = new Tensor(0.0, shape_o, 2);

    double min = -2 * radius;
    double max = 2 * radius;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    for (int i = 0; i < input->tensor_shape.size; i++)
    {
        input->data[i] = dis(gen);
    }

    radius_ *= radius_;

    for (int i = 0; i < output->tensor_shape.shape[0]; i++)
    {
        Tensor I = input->index(1, i);

        double x = I.data[0];
        double y = I.data[1];

        x *= x;
        y *= y;

        if (x + y < radius_)
        {
            output->data[i] = 1;
        }
        else
        {
            output->data[i] = 0;
        }
    }
}

// MNIST ##################################################

MNIST::MNIST(std::string path_, std::string type_) : path(path_), type(type_)
{
    std::cout << "MNIST::MNIST(std::string path_, std::string type_)" << std::endl;
    
    if (type == "train")
    {
        path += "train.csv";
        n_samples = 60000;
    }
    else if (type == "test")
    {
        path += "test.csv";
        n_samples = 10000;
    }
    else
    {
        std::cerr << "Type Error : type must be train or test" << std::endl;
    }

    int shape_i[] = {n_samples, 28, 28, 1};
    int shape_o[] = {n_samples, 1};

    input = new Tensor(shape_i, 4);
    output = new Tensor(shape_o, 2);

    std::string buffer;
    std::fstream fs;
    double data;

    fs.open(path, std::ios::in);

    getline(fs, buffer);

    for (int sample = 0; sample < n_samples; sample++)
    {
        // Label
        getline(fs, buffer, ',');

        std::stringstream rawdata1(buffer);
        rawdata1 >> data;

        if (rawdata1.fail())
        {
            std::cerr << "Data Processing Fail : String to Numeric" << std::endl;
        }
        else
        {
            output->data[sample] = data;
        }

        // Input
        for (int i = 0; i < 28 * 28; i++)
        {
            getline(fs, buffer, ',');

            std::stringstream rawdata2(buffer);
            rawdata2 >> data;

            if (rawdata2.fail())
            {
                std::cerr << "Data Processing Fail : String to Numeric" << std::endl;
            }
            else
            {
                int idx = sample * 28 * 28 + i;
                input->data[idx] = data;
            }
        }
    }

    if (fs.eof())
    {
        std::cout << "Data Processing Complete" << std::endl;
        fs.close();
    }
    else
    {
        std::cout << "Data Processing Not Complete" << std::endl;
        fs.close();
    }
}

void MNIST::visualize(int idx_)
{
    Tensor I = input->index(1, idx_);
    Tensor O = output->index(1, idx_);

    int shape_[] = {28, 28};
    I.reshape(shape_, 2);

    std::cout << "input :" << std::endl;

    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            int idx = i * 28 + j;

            if (I.data[idx] > 0)
            {
                std::cout << '*';
            }
            else
            {
                std::cout << ' ';
            }
        }

        std::cout << std::endl;
    }

    std::cout << "label : " << O.data[0] << std::endl;
}

// DataLoader #############################################

DataLoader::DataLoader(Dataset *dataset_, int batch_size_, bool shuffle_) : dataset(dataset_), batch_size(batch_size_), shuffle(shuffle_), verbose(false)
{
    if (verbose)
        std::cout << "DataLoader::DataLoader(Dataset *dataset_, int batch_size_, bool shuffle_)" << std::endl;

    batch_idx = new int[batch_size];
}

void DataLoader::batching()
{
    if (verbose)
        std::cout << "void DataLoader::batching()" << std::endl;

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
    if (verbose)
        std::cout << "Tensor DataLoader::input()" << std::endl;

    Tensor batch_input;

    for (int i = 0; i < batch_size; i++)
    {
        int idx = batch_idx[i];
        Tensor batch = dataset->input->index(1, idx);
        batch_input.append(batch);
    }

    return batch_input;
}

Tensor DataLoader::label()
{
    Tensor batch_label;

    for (int i = 0; i < batch_size; i++)
    {
        int idx = batch_idx[i];
        batch_label.append(dataset->output->index(1, idx));
    }

    return batch_label;
}