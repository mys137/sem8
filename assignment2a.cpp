#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <omp.h>

using namespace std;
// odd even bubble sort using the parallel programing
void parallel_bubble_sort_odd_even(int arr[], int n) {
    int phase, i, temp;
    for (phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {  // Even phase
            #pragma omp parallel for private(i, temp)
            for (i = 2; i < n; i += 2) {
                if (arr[i - 1] > arr[i]) {
                    temp = arr[i - 1];
                    arr[i - 1] = arr[i];
                    arr[i] = temp;
                }
            }
        } else {  // Odd phase
            #pragma omp parallel for private(i, temp)
            for (i = 1; i < n; i += 2) {
                if (arr[i - 1] > arr[i]) {
                    temp = arr[i - 1];
                    arr[i - 1] = arr[i];
                    arr[i] = temp;
                }
            }
        }
    }
}

// bubble sort for serial computation 
void bubble_sort(int arr[], int n) {
    int i, j;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

//merge function for the serial computation 
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];
    int i, j, k;

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

//merge sort for serial computation 
void merge_sort(int arr[], int l, int r) {
    if (l < r) {
        // find the middle element 
        int m = l + (r - l) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // merge the array 
                merge_sort(arr, l, m);
            }

            #pragma omp section
            {
                // merge the array
                merge_sort(arr, m + 1, r);
            }
        }

        merge(arr, l, m, r);
    }
}

// funtion to print the sorted array in serial computation 
void print_array(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// main funciton 
int main() {
    srand(time(NULL));

    
    int n = 1000;// numbr of elements

    int arr[n]; // unsorted array 

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000; // getting the random value in the array undr range 1 - 1000
    }

    double start_time, end_time;

    // printing the array
    cout<<"-----------------------------------------------------------------\n";
    cout << "Sequential Bubble Sort" << endl;
    start_time = omp_get_wtime();
    bubble_sort(arr, n);
    end_time = omp_get_wtime();
    cout << "Time taken: " << end_time - start_time << " seconds" << endl;

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }

    cout<<"-----------------------------------------------------------------\n";

    cout << "Parallel Bubble Sort" << endl;
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        parallel_bubble_sort_odd_even(arr, n);
    }
    end_time = omp_get_wtime();
    cout << "Time taken: " << end_time - start_time << " seconds" << endl;

    for (int i = 0; i < n; i++) {
       arr[i] = rand() % 10000;
}

cout<<"-----------------------------------------------------------------\n";
cout << "Sequential Merge Sort" << endl;
start_time = omp_get_wtime();
merge_sort(arr, 0, n - 1);
end_time = omp_get_wtime();
cout << "Time taken: " << end_time - start_time << " seconds" << endl;

for (int i = 0; i < n; i++) {
    arr[i] = rand() % 10000;
}

cout<<"-----------------------------------------------------------------\n";

cout << "Parallel Merge Sort" << endl;
start_time = omp_get_wtime();
#pragma omp parallel
{
    #pragma omp single nowait
    {
        merge_sort(arr, 0, n - 1);
    }
}
end_time = omp_get_wtime();
cout << "Time taken: " << end_time - start_time<<setprecision(20) <<" seconds" << endl;

cout<<"-----------------------------------------------------------------\n";

return 0;
}

