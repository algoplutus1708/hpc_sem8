#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

void parallelBubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int phase = 0; phase < n; phase++)
    {
        if (phase % 2 == 0)
        {
#pragma omp parallel for
            for (int i = 1; i < n - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                    swap(arr[i], arr[i + 1]);
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < n - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                    swap(arr[i], arr[i + 1]);
            }
        }
    }
}

void merge(vector<int> &arr, vector<int> &temp, int left, int mid, int right)
{
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i <= mid)
        temp[k++] = arr[i++];
    while (j <= right)
        temp[k++] = arr[j++];
    for (i = left; i <= right; i++)
        arr[i] = temp[i];
}

void parallelMergeSortHelper(vector<int> &arr, vector<int> &temp, int left, int right, int depth)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;
        if (depth < 4)
        {
#pragma omp task shared(arr, temp)
            parallelMergeSortHelper(arr, temp, left, mid, depth + 1);

#pragma omp task shared(arr, temp)
            parallelMergeSortHelper(arr, temp, mid + 1, right, depth + 1);

#pragma omp taskwait
        }
        else
        {
            parallelMergeSortHelper(arr, temp, left, mid, depth + 1);
            parallelMergeSortHelper(arr, temp, mid + 1, right, depth + 1);
        }
        merge(arr, temp, left, mid, right);
    }
}

void parallelMergeSort(vector<int> &arr)
{
    vector<int> temp(arr.size());
#pragma omp parallel
    {
#pragma omp single
        parallelMergeSortHelper(arr, temp, 0, arr.size() - 1, 0);
    }
}

int main()
{
    vector<int> arr = {5, 2, 9, 1, 5, 6, 3, 8};
    parallelBubbleSort(arr);
    return 0;
}