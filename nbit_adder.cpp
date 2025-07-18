#include<bits/stdc++.h>
#include"mli.h"

using namespace std;
using Mat = Matrix<float>;


constexpr int bits = 4;



struct Gaussian {
    double operator()(const double x) const {
        return exp(-x*x);
    }
    [[nodiscard]] double derivative(const double x) const {
        return -2*x*(*this)(x);
    }
};



int main() {
    srand(time(0));
    size_t n = 1 << bits;
    size_t rows = n*n;
    Mat tin(rows,2*bits), tout(rows,bits+1);
    for (size_t i = 0; i < rows; ++i) {
        size_t x = i / n;
        size_t y = i % n;
        size_t z = x + y;
        for (size_t j = 0; j < bits; ++j) {
            tin[i][j] = (x >> j) & 1;
            tin[i][j + bits] = (y >> j) & 1;
            tout[i][j] = (z >> j) & 1;
        }
        tout[i][bits] = (z >> bits) & 1;
    }
    const std::vector<size_t> layers{2*bits,4*bits,bits+1};
    Neural<float,Gaussian> nn(layers);
    Neural<float,Gaussian> grad(layers);

    cout << nn.cost(tin,tout) << endl;
    for (int i = 0; i < 3e4; i++) {
        constexpr float rate = 1;
        nn.gradients(grad,tin,tout);
        nn.apply_gradients(grad,rate);
        if (i%500==0)
            cout << i << ": c = " << nn.cost(tin,tout) << endl;
    }
    cout << nn.cost(tin,tout) << endl << endl;
    for (size_t i = 0; i < tin.n; ++i) {
        nn.activation[0] = tin.row(i);
        nn.forward();


        auto vec = nn.activation.back()[0];
        vector<int> num1(tin[i].begin(),tin[i].begin()+bits);
        vector<int> num2(tin[i].begin()+bits,tin[i].end());
        int pw = 1; long first=0, second=0, third = 0;
        for (int j : num1) {
            first += pw*j;
            pw *= 2;
        }
        pw = 1;
        for (int j : num2) {
            second += pw*j;
            pw *= 2;
        }
        pw = 1;
        for (auto j : vec) {
            third += pw*(j > 0.5 ? 1 : 0);
            pw*=2;
        }
        if (first + second != third)
            cout << first << " + " << second << " = " << third << endl;
    }
}
