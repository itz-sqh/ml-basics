#include<bits/stdc++.h>
#include"mli.h"

using namespace std;
using Mat = Matrix<>;





//xor gate
Mat tin = {
    {0,0},
        {0,1},
        {1,0},
        {1,1},
};
Mat tout = {
    {0},
    {1},
    {1},
    {0},
};

struct Gaussian {
    double operator()(const double x) const {
        return exp(-x*x);
    }
    double derivative(const double x) const {
        return -2*x*(*this)(x);
    }
};



int main() {
    srand(time(0));

    const std::vector<size_t> layers{2,2,1};
    Neural<double, Gaussian> nn1(layers);
    Neural<double, Gaussian> nn2(layers);
    Neural<double, Gaussian> grad1(layers);
    Neural<double, Gaussian> grad2(layers);

    cout << nn1.cost(tin,tout) << endl;
    for (int i = 0; i < 20000; i++) {
        constexpr double rate = 1e-1;
        nn1.gradients(grad1,tin,tout);
        nn1.apply_gradients(grad1,rate);
    }
    cout << nn1.cost(tin,tout) << endl << endl;
    for (size_t i = 0; i < tin.n; ++i) {
        nn1.activation[0] = tin.row(i);
        nn1.forward();
        double result = nn1.activation.back()[0][0];
        int x = tin[i][0];
        int y = tin[i][1];
        cout << x << " ^ " << y << " = " << result << endl;
    }

    cout << endl << nn2.cost(tin,tout) << endl;
    for (int i = 0; i < 20000; i++) {
        constexpr double rate = 1e-1;
        nn2.gradients_naive(grad2,rate,tin,tout);
        nn2.apply_gradients(grad2,rate);
    }
    cout << nn2.cost(tin,tout) << endl;
    for (size_t i = 0; i < tin.n; ++i) {
        nn2.activation[0] = tin.row(i);
        nn2.forward();
        double result = nn2.activation.back()[0][0];
        int x = tin[i][0];
        int y = tin[i][1];
        cout << x << " ^ " << y << " = " << result << endl;
    }

}
