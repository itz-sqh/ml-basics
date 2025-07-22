#include<iostream>
#include "../neural.h"

using Mat = Matrix<float>;
using NN = Neural<float,Gaussian<float>>;


constexpr int bits = 4;
constexpr size_t n = 1 << bits;
constexpr size_t rows = n * n;


void data_init(Mat& tin, Mat& tout) {

    for (size_t i = 0; i < rows; ++i) {
        const size_t x = i / n;
        const size_t y = i % n;
        const size_t z = x + y;

        for (size_t j = 0; j < bits; ++j) {
            tin[i][j] = (x >> j) & 1;
            tin[i][j + bits] = (y >> j) & 1;

            tout[i][j] = (z >> j) & 1;
        }

        tout[i][bits] = (z >> bits) & 1;
    }

}



int main() {

    srand(time(nullptr));

    Mat tin(rows, 2*bits), tout(rows, bits+1);
    data_init(tin, tout);

    const std::vector<size_t> layers{2*bits, bits*bits, bits+1};
    NN nn(layers);
    NN grad(layers);

    nn.random_init(-0.2f,0.2f);

    constexpr int iteration_count = 1e3;

    std::cout << "Initial cost = " << nn.cost(tin, tout) << std::endl;

    for (int i = 0; i < iteration_count; i++) {

        constexpr double rate = 1;
        constexpr int batch = 50;
        constexpr int epochs = 5;
        constexpr bool shuffle = true;

        nn.batch_process(grad,tin,tout,rate,batch,epochs,shuffle);

        if (i % 500 == 0) {
            std::cout << i << ": cost = " << nn.cost(tin, tout) << std::endl;
        }
    }

    std::cout << "Final cost = " << nn.cost(tin,tout) << std::endl << std::endl;


    bool ok = true;
    for (size_t i = 0; i < tin.n; ++i) {
        nn.activation[0] = tin.row(i);
        nn.forward();

        auto prediction = nn.activation.back()[0];

        int first = 0;
        for (size_t j = 0; j < bits; ++j) {
            first |= (tin[i][j] > 0.5f) << j;
        }

        int second = 0;
        for (size_t j = 0; j < bits; ++j) {
            second |= (tin[i][bits + j] > 0.5f) << j;
        }

        int predict = 0;
        for (size_t j = 0; j < prediction.size(); ++j) {
            predict |= (prediction[j] > 0.5f) << j;
        }
        if (first + second != predict) {
            std::cout << first << " + " << second << " = " << predict << std::endl;
            ok = false;
        }

        if (ok) std::cout << "OK";
    }
}