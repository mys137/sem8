#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

double linearRegression(vector<double>& x, vector<double>& y){
    double x_mean = 0, y_mean = 0;
    int n = x.size();

    #pragma omp parallel for reduction(+:x_mean, y_mean)
        for(int i=0;i<n;i++){
            x_mean += x[i];
            y_mean += y[i];
        }
    x_mean /= n;
    y_mean /= n;

    double nume = 0, deno = 0;
    #pragma omp parallel for reduction(+: nume, deno)
        for(int i=0;i<n;i++){
            nume+=(x[i]-x_mean) * (y[i] - y_mean);
            deno+=(x[i]-x_mean) * (x[i] - x_mean);
        }
    

    return nume/deno;
}
int main(){
    vector<double> x = {1,2,6,5};
    vector<double> y = {3,6,15,8};

    double result = linearRegression(x,y);
    cout<< " the slope of the linear regression line is: "<<result<<endl;
    return 0;
}
