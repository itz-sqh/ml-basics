#include<iostream>
#include "../neural.h"


using Mat = Matrix<double>;
using NN = Neural<double,Gaussian<double>>;


const Mat tin = {
    {0,0},
    {0,1},
    {1,0},
    {1,1},
};
const Mat tout = {
    {0},
    {1},
    {1},
    {0},
};


int main() {
    srand(time(nullptr));

    const std::vector<size_t> layers{2, 2, 1};

    NN nn(layers);
    NN grad(layers);

    nn.random_init(-0.2,0.2);

    constexpr int iteration_count = 1e4;

    std::cout << "Initial cost = " << nn.cost(tin, tout) << std::endl;

    for (int i = 0; i < iteration_count; i++) {
        constexpr double rate = 1e-1;

        nn.gradients(grad, tin, tout);
        nn.apply_gradients(grad, rate);
    }

    std::cout << "Final cost = " << nn.cost(tin,tout) << std::endl << std::endl;

    for (size_t i = 0; i < tin.n; ++i) {
        nn.activation[0] = tin.row(i);
        nn.forward();

        const double res = nn.activation.back()[0][0];

        const int x = tin[i][0];
        const int y = tin[i][1];

        std::cout << x << " ^ " << y << " = " << res << std::endl;
    }

}