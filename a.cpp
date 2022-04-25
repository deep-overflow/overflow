#include <iostream>
using namespace std;

int main()
{
    int *a = new int[10];

    cout << sizeof(*a) << endl;
    cout << sizeof(int) << endl;
    cout << sizeof(*a) / sizeof(int) << endl;
}