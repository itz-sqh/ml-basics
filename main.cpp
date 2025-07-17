#include<bits/stdc++.h>
#include"mli.h"

using namespace std;
using Mat = Matrix<>;
using NN = Neural<>;


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



int main() {
    srand(420);

    const std::vector<size_t> layers{2,2,1};
    NN nn(layers);
    NN grad(layers);

    cout << nn.cost(tin,tout) << endl;
    for (int i = 0; i < 30000; i++) {
        constexpr double rate = 1e-1;
        constexpr double eps = 1e-1;
        nn.gradients_naive(grad,eps,tin,tout);
        nn.apply_gradients(grad,rate);
    }
    cout << nn.cost(tin,tout) << endl;
    cout << nn;
    for (size_t i = 0; i < tin.n; ++i) {
        nn.activation[0] = tin.row(i);
        nn.forward();
        double result = nn.activation.back()[0][0];
        int x = tin[i][0];
        int y = tin[i][1];
        cout << x << " ^ " << y << " = " << result << endl;
    }
}
