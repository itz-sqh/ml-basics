#include<bits/stdc++.h>

using namespace std;

vector<pair<double,double>> train = {
    {0,1},
    {1,3},
    {2,5},
    {3,7},
    {4,9},
};

int train_count = size(train);

double rand_dbl() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

double cost(double w, double b) {
    double res = 0;
    for (int i = 0; i < train_count; i++) {
        double x = train[i].first;
        double y = x*w+b;
        double d = y-train[i].second;
        res += d*d;
    }
    return res/train_count;
}

void grad(double w, double b, double eps, double& dw, double& db) {
    double c = cost(w,b);
    dw = (cost(w+eps,b)-c)/eps;
    db = (cost(w,b+eps)-c)/eps;
}
void learn(int iteration_count, double& w, double& b,double eps,double rate,double& dw, double& db) {
    for (int i = 0; i < iteration_count; i++) {
        grad(w,b,eps,dw,db);
        w-=rate*dw;
        b-=rate*db;
    }
}

int main() {
    srand(time(0));
    double w = rand_dbl();
    double b = rand_dbl();
    double dw,db;
    double eps = 1e-3;
    double rate = 1e-3;
    cout << cost(w,b) << endl;
    learn(10000,w,b,eps,rate,dw,db);
    cout << cost(w,b) << endl;

}
