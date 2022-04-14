#include "nn.h"

// Linear #################################################

Linear::Linear(int in_features_, int out_features_)
{
    int shape_[] = {out_features_, in_features_};

    params.init(1.0, shape_, 2);
}

Tensor *Linear::operator()(Tensor *input_)
{
    if (output == NULL) {
        output = new Tensor;
    }

    *output = dot(params, *input_);
    output->func = this;

    input = input_;

    return output;
}

void Linear::backward()
{
    // 여기서부터 다시 해보자.
}

void Linear::print()
{
    std::cout << "===== Linear class =====" << std::endl;
    params.print();
}

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################

// Shape ##################################################
