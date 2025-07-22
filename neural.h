#pragma once
#include"neuralmath.h"




template<typename ntype = float, typename Activation = Sigmoid<ntype>>
struct Neural {
    size_t n;

    std::vector<Matrix<ntype>> weight;
    std::vector<Matrix<ntype>> bias;
    std::vector<Matrix<ntype>> activation;

    Activation act;

    explicit Neural(const std::vector<size_t>& layers);

    constexpr void random_init(ntype low, ntype high);
    constexpr void xavier_init();

    constexpr void forward();

    // TODO template for loss functions
    // for now cost supports MSE only
    ntype cost(const Matrix<ntype>& tin, const Matrix<ntype>& tout);

    constexpr void apply_gradients(Neural& grad, ntype rate);

    // finite differences
    constexpr void gradients_naive(Neural& grad, ntype eps, const Matrix<ntype>& tin, const Matrix<ntype>& tout);

    // gradient descent
    void gradients(Neural& grad, const Matrix<ntype>& tin, const Matrix<ntype>& tout);

    // stochastic gradient descent
    void batch_process(
        Neural& grad,
        const Matrix<ntype>& tin,
        const Matrix<ntype>& tout,
        ntype rate,
        size_t batch = 32,
        size_t epochs = 3, bool shuffle = true);



    constexpr void zero();

    constexpr void to_stream(std::ostream& os,
                        int precision  = 4,
                        int col_width  = 10) const;

};

template<typename ntype, typename Activation>
Neural<ntype,Activation>::Neural(const std::vector<size_t>& layers) : n(layers.size()-1) {
    assert(!layers.empty() && "Neural must have at least input and output layers");

    weight.resize(n);
    bias.resize(n);
    activation.resize(n+1);

    for (size_t i = 0; i < n; i++) {
        weight[i] = Matrix<ntype>(layers[i],layers[i+1]);
        bias[i] = Matrix<ntype>(1,layers[i+1]);
    }

    for (size_t i = 0; i < n + 1; i++) {
        activation[i] = Matrix<ntype>(1,layers[i]);
    }
}

template<typename ntype, typename Activation>
constexpr void Neural<ntype,Activation>::random_init(ntype low, ntype high) {
    for (size_t i = 0; i < n; i++) {
        weight[i].rand_init(low,high);
        bias[i].rand_init(low,high);
    }
}

template<typename ntype, typename Activation>
constexpr void Neural<ntype,Activation>::xavier_init() {
    for (size_t i = 0; i < n; i++) {
        float range = std::sqrt(6.0f/(weight[i].n + weight[i].m));
        weight[i].rand_init(-range,range);
    }
}

template<typename ntype, typename Activation>
constexpr void Neural<ntype,Activation>::forward() {
    for (size_t i = 0; i < n; i++) {
        activation[i+1] = activation[i] * weight[i];
        activation[i+1] += bias[i];
        activation[i+1].apply(act);
    }
}

template<typename ntype, typename Activation>
ntype Neural<ntype,Activation>::cost(const Matrix<ntype> & tin, const Matrix<ntype>& tout) {
    assert(tin.n == tout.n && "Input and output must have the same number of samples");
    assert(tin.m == activation[0].m && "Input size must match neural input size");
    assert(tout.m == activation.back().m && "Output size must match neural output size");

    size_t p = tin.n;
    ntype res = 0;

    for (size_t i = 0; i < p; i++) {
        auto x = tin.row(i);
        auto y = tout.row(i);

        activation[0] = x;
        forward();

        size_t q = tout.m;
        for (size_t j = 0; j < q; j++) {
            ntype d = activation.back()[0][j] - y[0][j];
            res += d*d;
        }
    }
    return res/p;
}

template<typename ntype, typename Activation>
constexpr void Neural<ntype,Activation>::gradients_naive(Neural& grad, ntype eps, const Matrix<ntype>& tin, const Matrix<ntype>& tout) {
    assert(tin.n == tout.n && "Input and output must have the same number of samples");
    assert(tin.m == activation[0].m && "Input size must match neural input size");
    assert(tout.m == activation.back().m && "Output size must match neural output size");

    ntype saved;
    ntype c = this->cost(tin,tout);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < weight[i].n; j++) {
            for (size_t k = 0; k < weight[i].m; k++) {
                saved = weight[i][j][k];

                weight[i][j][k] += eps;
                grad.weight[i][j][k] = (this->cost(tin,tout)-c)/eps;

                weight[i][j][k] = saved;
            }
        }
        for (size_t j = 0; j < bias[i].n; j++) {
            for (size_t k = 0; k < bias[i].m; k++) {
                saved = bias[i][j][k];

                bias[i][j][k]+=eps;
                grad.bias[i][j][k] = (this->cost(tin,tout)-c)/eps;

                bias[i][j][k] = saved;
            }
        }
    }
}

template<typename ntype, typename Activation>
void Neural<ntype,Activation>::gradients(Neural& grad, const Matrix<ntype>& tin, const Matrix<ntype>& tout) {
    const size_t p = tin.n;
    grad.zero();

    std::vector<Matrix<ntype>> a(n+1);
    std::vector<Matrix<ntype>> z(n);
    std::vector<Matrix<ntype>> d(n);

    for (size_t i = 0; i < p; i++) {
        Matrix<ntype> x = tin.row(i);
        Matrix<ntype> y = tout.row(i);

        a[0] = x;
        // forward
        for (size_t j = 0; j < n; j++) {
            z[j] = a[j]*weight[j]+bias[j];
            a[j+1] = z[j];
            a[j+1].apply(act);
        }

        d[n-1] = a[n] - y;
        for (size_t j = 0; j < d[n-1].m; j++)
            d[n-1][0][j] *= act.derivative(z[n-1][0][j]);

        for (int l = n - 2; l>=0; l--) {
            d[l] = d[l+1] * weight[l+1].transpose();
            for (size_t j = 0; j < d[l].m; j++) {
                d[l][0][j] *= act.derivative(z[l][0][j]);
            }
        }

        for (size_t l = 0; l < n; l++) {
            grad.bias[l] += d[l];
            grad.weight[l] += a[l].transpose() * d[l];
        }
    }
    for (size_t i = 0; i < n; i++) {
        grad.weight[i] *= static_cast<ntype>(1.0/p);
        grad.bias[i] *= static_cast<ntype>(1.0/p);
    }
}

template<typename ntype, typename Activation>
void Neural<ntype,Activation>::batch_process(Neural& grad, const Matrix<ntype>& tin, const Matrix<ntype>& tout, ntype rate, size_t batch, size_t epochs, bool shuffle) {
    size_t n = tin.n;

    for (int e = 0; e < epochs; e++) {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(),idx.end(),0);

        if (shuffle) std::ranges::shuffle(idx, std::mt19937(std::random_device()()));

        for (size_t i = 0; i < n; i+=batch) {
            size_t r = std::min(i+batch,n);
            size_t len = r - i;

            Matrix<ntype> in(len,tin.m);
            Matrix<ntype> out(len,tout.m);

            for (size_t j = 0; j < len; j++) {
                size_t id = idx[i+j];
                std::copy(tin[id].begin(),tin[id].end(),in[j].begin());
                std::copy(tout[id].begin(),tout[id].end(),out[j].begin());
            }
            gradients(grad,in,out);
            apply_gradients(grad, rate);
        }
    }
}

template<typename ntype, typename Activation>
constexpr void Neural<ntype,Activation>::apply_gradients(Neural& grad, ntype rate) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < weight[i].n; j++) {
            for (size_t k = 0; k < weight[i].m; k++) {
                weight[i][j][k] -= rate*grad.weight[i][j][k];
            }
        }
        for (size_t j = 0; j < bias[i].n; j++) {
            for (size_t k = 0; k < bias[i].m; k++) {
                bias[i][j][k] -= rate*grad.bias[i][j][k];
            }
        }
    }
}

template<typename ntype, typename Activation>
constexpr void Neural<ntype,Activation>::zero() {
    for (size_t i = 0; i < n; i++) {
        weight[i].fill(0);
        bias[i].fill(0);
        activation[i].fill(0);
    }
    activation[n].fill(0);
}

template<typename ntype, typename Activation>
constexpr void Neural<ntype,Activation>::to_stream(
    std::ostream& os,
    int precision,
    int col_width) const
{
    os << "nn = (\n";

    for (size_t i = 0; i < weight.size(); ++i) {
        std::string label = "  w" + std::to_string(i) + " = ";
        os << label;

        weight[i].to_stream(os, precision, col_width, static_cast<int>(label.size()));
        os << "\n";
    }

    for (size_t i = 0; i < bias.size(); ++i) {
        std::string label = "  b" + std::to_string(i) + " = ";
        os << label;

        bias[i].to_stream(os, precision, col_width, static_cast<int>(label.size()));
        os << "\n";
    }

    os << ")\n";
}

template<typename ntype, typename Activation> constexpr std::ostream& operator << (std::ostream& st, const Neural<ntype,Activation>& neural) { neural.to_stream(st); return st; }
