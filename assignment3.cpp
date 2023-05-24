#include <bits/stdc++.h>
#include <ctime>
#include <omp.h>

using namespace std;

// Parallel reduction to find min value
int parallel_min( vector<int>& data)
{
    int min_value = data[0];
    #pragma omp parallel for reduction(min:min_value)
    for (int i = 1; i < data.size(); ++i)
        if (data[i] < min_value)
            min_value = data[i];

    return min_value;
}

// Parallel reduction to find max value
int  parallel_max( vector<int>& data)
{
    int max_value = data[0];
    #pragma omp parallel for reduction(max:max_value)
    for (int i = 1; i < data.size(); ++i)
        if (data[i] > max_value)
            max_value = data[i];
    return max_value;
}

// Parallel reduction to find sum
int parallel_sum( vector<int>& data)
{
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < data.size(); ++i)
        sum += data[i];

    return sum;
}

// Parallel reduction to find average
 double parallel_average(vector<int>& data)
{
    int sum = parallel_sum(data);
    double average = static_cast<double>(sum) / data.size();
    return average;
}

int serial_min( vector<int>& data)
{
    int min_value = data[0];
    for (int i = 1; i < data.size(); ++i)
        if (data[i] < min_value)
            min_value = data[i];

    return min_value;
}

// Parallel reduction to find max value
int  serial_max( vector<int>& data)
{
    int max_value = data[0];
    for (int i = 1; i < data.size(); ++i)
        if (data[i] > max_value)
            max_value = data[i];
    return max_value;
}

// Parallel reduction to find sum
int serial_sum( vector<int>& data)
{
    int sum = 0;
    for (int i = 0; i < data.size(); ++i)
        sum += data[i];

    return sum;
}

// Parallel reduction to find average
 double serial_average(vector<int>& data)
{
    int sum = parallel_sum(data);
    double average = static_cast<double>(sum) / data.size();
    return average;
}


int main() {
    // Ask user for the size of the vector
    int size;
    cout << "Enter the size of the vector: ";
    cin >> size;

    // Ask user for the values of the vector
    vector<int> data(size);
    cout << "Enter the values of the vector:" << endl;
    for (int i = 0; i < size; ++i) {
        data[i] = rand()%20;
    }

    // Find min, max, sum and average using parallel reduction
    auto start_time = omp_get_wtime();

    int min_value = parallel_min(data);
    int max_value = parallel_max(data);
    int sum = parallel_sum(data);
    double average = parallel_average(data);

    auto end_time = omp_get_wtime();

    // Print results and timing information
    cout << "Min value: " << min_value << endl;
    cout << "Max value: " << max_value << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << average << endl;
    auto duration_ms = end_time - start_time;
    cout << "Time taken by parallel: " << duration_ms << "sec" << endl;


    cout<<"----------------------------------------------------------\n";

    auto sstart_time = omp_get_wtime();

    int smin_value = serial_min(data);
    int smax_value = serial_max(data);
    int ssum = serial_sum(data);
    double saverage = serial_average(data);

    auto send_time = omp_get_wtime();

    // Print results and timing information
    cout << "Min value: " << smin_value << endl;
    cout << "Max value: " << smax_value << endl;
    cout << "Sum: " << ssum << endl;
    cout << "Average: " << saverage << endl;
    auto sduration_ms = send_time - sstart_time;
    cout << "Time taken by serial: " << sduration_ms << "sec" << endl;

    return 0;
}
