#include<bits/stdc++.h>
#include"mli.h"

using namespace std;
using Mat = Matrix<>;





Mat tin = {
    {0,0},
        {0,1},
        {1,0},
        {1,1},
};
Mat tout = {
    {0,0},
    {0,1},
    {0,1},
    {1,0},
};

struct Gaussian {
    double operator()(const double x) const {
        return exp(-x*x);
    }
    [[nodiscard]] double derivative(const double x) const {
        return -2*x*(*this)(x);
    }
};



int main() {
    srand(time(nullptr));

    const std::vector<size_t> layers{2,2,2};
    Neural nn(layers);
    Neural grad(layers);

    cout << nn.cost(tin,tout) << endl;
    for (int i = 0; i < 50000; i++) {
        constexpr double rate = 1e-1;
        nn.gradients(grad,tin,tout);
        nn.apply_gradients(grad,rate);
    }
    cout << nn.cost(tin,tout) << endl << endl;
    for (size_t i = 0; i < tin.n; ++i) {
        nn.activation[0] = tin.row(i);
        nn.forward();
        auto vec = nn.activation.back()[0];
        int x = static_cast<int>(tin[i][0]);
        int y = static_cast<int>(tin[i][1]);
        cout << x << " + " << y << " = " << vec[0] << " " << vec[1] << endl;
    }


}
