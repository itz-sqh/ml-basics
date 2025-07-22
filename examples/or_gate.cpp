#include<iostream>
#include<vector>
#include<cmath>

std::vector<std::vector<double>> train = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1},
};

int train_count = size(train);

double rand_dbl() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double cost(const double w1, const double w2, const double b) {
    double res = 0;

    for (int i = 0; i < train_count; i++) {
        const double x1 = train[i][0];
        const double x2 = train[i][1];

        const double y = sigmoid(x1 * w1 + x2 * w2 + b);

        const double d = y-train[i][2];
        res += d*d;
    }

    return res/train_count;
}

void grad(const double w1, const double w2, const double b, const double eps, double& dw1, double& dw2, double& db) {
    double c = cost(w1,w2,b);
    dw1 = (cost(w1 + eps, w2, b) - c) / eps;
    dw2 = (cost(w1, w2 + eps, b) - c) / eps;
    db = (cost(w1, w2, b + eps) - c) / eps;
}

void learn(const int iteration_count, double& w1, double& w2, double& b, double eps, double rate, double& dw1, double& dw2, double& db) {
    for (int i = 0; i < iteration_count; i++) {
        grad(w1, w2, b, eps, dw1, dw2, db);
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }
}

int main() {
    srand(time(nullptr));

    double w1 = rand_dbl();
    double w2 = rand_dbl();
    double b = rand_dbl();

    double dw1,dw2,db;

    constexpr double eps = 1e-1;
    constexpr double rate = 1e-1;

    std::cout << "Initial cost = " << cost(w1, w2, b) << std::endl;

    learn(1e5, w1, w2, b, eps, rate, dw1, dw2, db);

    std::cout << "Final cost = " << cost(w1, w2, b) << std::endl;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << i << " | " << j << " = " << sigmoid(i * w1 + j * w2 + b) << std::endl;
        }
    }

}
