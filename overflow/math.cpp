#include "math.h"

double exp_(int x)
{
    double value = 1;

    for (int i = 0; i < x; i++)
    {
        value *= M_E;
    }

    return value;
}

double normal_(double x, double mu, double sigma)
{
    double value = 1 / (sigma * sqrt_(2 * M_PI));
    double k = -(x - mu) * (x - mu) / (2 * sigma * sigma);
    value *= exp_(k);
    
    return value;
}

double sqrt_(double x)
{
    double s = x / 2;
    double t = 0;

    while (s != t)
    {
        t = s;
        s = ((x / t) + t) / 2;
    }
    
    return s;
}
