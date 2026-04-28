#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>

using namespace std;

int main()
{
    int size = 10000000;
    vector<int> data(size, 1);

    int minVal = numeric_limits<int>::max();
    int maxVal = numeric_limits<int>::min();
    long long sum = 0;

#pragma omp parallel reduction(min : minVal) reduction(max : maxVal) reduction(+ : sum)
    {
#pragma omp for
        for (int i = 0; i < size; i++)
        {
            if (data[i] < minVal)
                minVal = data[i];
            if (data[i] > maxVal)
                maxVal = data[i];
            sum += data[i];
        }
    }

    double average = static_cast<double>(sum) / size;

    cout << "Min: " << minVal << "\nMax: " << maxVal << "\nSum: " << sum << "\nAvg: " << average << endl;
    return 0;
}