#pragma once
#include <vector>
#include<cassert>
#include<iomanip>

template<typename mtype = double>
mtype rand_val() {
    return static_cast<mtype>(rand())/static_cast<mtype>(RAND_MAX);
}

template<typename atype>
struct Sigmoid {
    atype operator()(atype x) const {
        return 1.0/(1.0+std::exp(-x));
    }
    atype derivative(atype x) const {
        atype s = (*this)(x);
        return s*(1-s);
    }
};

template<typename mtype = double>
struct Matrix {
    constexpr Matrix() : n(0), m(0) {}
    constexpr Matrix(size_t n, size_t m, bool identity = false) : n(n),m(m), a(n,std::vector<mtype>(m,0)) {
        if (identity) {
            assert(!(identity && n!=m) && "Identity Matrix must be a square Matrix");
            for (size_t i = 0; i < n; i++) {
                a[i][i] = 1;
            }
        }
    }
    constexpr Matrix(std::initializer_list<std::initializer_list<mtype>> init) : n(init.size()), m(init.begin()->size()), a(n,std::vector<mtype>(m)) {
        size_t i = 0;
        for (const auto& row : init) {
            assert(row.size() == m && "All rows must be the same size");
            size_t j = 0;
            for (const auto& x : row) {
                a[i][j] = x;
                j++;
            }
            i++;
        }
    }
    size_t n,m;
    std::vector<std::vector<mtype>> a;

    constexpr Matrix operator+=(const Matrix& other) {
        assert(n == other.n && m == other.m && "Matrices must be the same size");
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                a[i][j] = a[i][j] + other.a[i][j];
            }
        }
        return *this;
    }
    constexpr Matrix operator+(const Matrix& other) const {
        return Matrix(*this) += other;
    }
    constexpr Matrix operator-=(const Matrix& other) {
        assert(n == other.n && m == other.m && "Matrices must be the same size");
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                a[i][j] = a[i][j] - other.a[i][j];
            }
        }
        return *this;
    }
    constexpr Matrix operator-(const Matrix& other) const {
        return Matrix(*this) -= other;
    }
    constexpr Matrix operator*(const Matrix& other) {
        assert(m == other.n && "Inner sizes must be equal");
        Matrix res(n,other.m);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < other.m; j++)
                for (size_t k = 0; k < m; k++)
                    res[i][j] += a[i][k] * other.a[k][j];
        return res;
    }
    constexpr Matrix operator*=(const Matrix& other) {
        *this = *this * other;
        return *this;
    }
    constexpr Matrix operator*(const std::vector<mtype>& vec) const {
        assert(m == vec.size() && "Matrix columns must match vector size");
        Matrix res(n,1);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < m; j++)
                res[i][0] += a[i][j] * vec[j];
        return res;
    }
    constexpr Matrix operator*=(const std::vector<mtype>& vec) {
        *this = *this * vec;
        return *this;
    }
    constexpr Matrix operator*(mtype val) {
        Matrix res(n,m);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < m; j++)
                res[i][j] = a[i][j] * val;
        return res;
    }
    constexpr Matrix operator*=(mtype val) {
        *this = *this * val;
        return *this;
    }

    std::vector<mtype>& operator[] (size_t i) {
        assert(i < n && "Index out of bounds");
        return a[i];
    }
    const std::vector<mtype>& operator[] (size_t i) const {
        assert(i < n && "Index out of bounds");
        return a[i];
    }
    constexpr bool operator==(const Matrix& other) const {
        assert(n == other.n && m == other.m && "Matrices must have same dimensions");
        return a == other.a;
    }
    constexpr bool operator!=(const Matrix& other) const {
        assert(n == other.n && m == other.m && "Matrices must have same dimensions");
        return a != other.a;
    }
    mtype& at(size_t i, size_t j) {
        assert(i<n && j < m && "Index out of bounds");
        return a[i][j];
    }
    const mtype& at(size_t i, size_t j) const {
        assert(i<n && j < m && "Index out of bounds");
        return a[i][j];
    }
    constexpr Matrix row(size_t r) const {
        assert(r < n && "Row index out of bounds");
        Matrix res(1,m);
        for (size_t i = 0; i < m; i++)
            res[0][i] = a[r][i];
        return res;
    }
    constexpr Matrix col(size_t c) const {
        assert(c < m && "Column index out of bounds");
        Matrix res(n, 1);
        for (size_t i = 0; i < n; ++i)
            res[i][0] = a[i][c];
        return res;
    }
    constexpr Matrix subMatrix(size_t r1, size_t c1, size_t r2, size_t c2) const {
        assert(r2 < n && c2 < m && r1 <= r2 && c1 <= c2);
        Matrix res(r2-r1+1, c2-c1+1);
        for (size_t i = r1; i <= r2; ++i)
            for (size_t j = c1; j <= c2; ++j)
                res.a[i-r1][j-c1] = a[i][j];
        return res;
    }
    constexpr Matrix transpose() const {
        Matrix res(m,n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                res.a[j][i] = a[i][j];
        return res;
    }
    constexpr void fill(mtype val) {
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < m; j++)
                a[i][j] = val;
    }
    template<typename ftype>
    void apply(ftype func) {
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < m; j++)
                a[i][j] = func(a[i][j]);
    }
    void rand(mtype min = {0}, mtype max = {1}) {
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < m; j++)
                a[i][j] = rand_val<mtype>()*(max-min)+min;
    }
    constexpr void from_stream(std::istream& is) {
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < m; j++)
                is >> a[i][j];
    }
    constexpr void to_stream(std::ostream& os,
               int precision,
               int col_width,
               int offset = 0) const
    {
        os << std::fixed << std::setprecision(precision)
           << "(\n";
        std::string pad(offset, ' ');
        for (size_t i = 0; i < n; ++i) {
            os << pad;
            for (size_t j = 0; j < m; ++j) {
                os << std::setw(col_width) << a[i][j];
                if (j + 1 < m) os << std::string(col_width / 2, ' ');
            }
            os << "\n";
        }
        os << pad << ")";
    }
};
template<typename mtype = double> constexpr std::istream& operator >> (std::istream& st, Matrix<mtype>& m) { m.from_stream(st); return st; }
template<typename mtype = double> constexpr std::ostream& operator << (std::ostream& st, const Matrix<mtype>& m) { m.to_stream(st); return st; }




template<typename ntype = double, typename Activation = Sigmoid<ntype>>
struct Neural {

    size_t n;
    Activation act;
    std::vector<Matrix<ntype>> weight;
    std::vector<Matrix<ntype>> bias;
    std::vector<Matrix<ntype>> activation;

    Neural(const std::vector<size_t>& layers, bool random_init = true) : n(layers.size()-1) {
        assert(layers.size() >= 1 && "Neural must have at least input and output layers");

        weight.resize(n);
        bias.resize(n);
        activation.resize(n+1);
        for (size_t i = 0; i < n; i++) {
            weight[i] = Matrix<ntype>(layers[i],layers[i+1]);
            bias[i] = Matrix<ntype>(1,layers[i+1]);
            if (random_init) {
                weight[i].rand();
                bias[i].rand();
            }
        }
        for (size_t i = 0; i < n + 1; i++) {
            activation[i] = Matrix<ntype>(1,layers[i]);
        }
    }
    void forward() {
        for (size_t i = 0; i < n; i++) {
            activation[i+1] = activation[i] * weight[i];
            activation[i+1] += bias[i];
            activation[i+1].apply(act);
        }
    }
    ntype cost(const Matrix<ntype> & tin, const Matrix<ntype>& tout) {
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
    void gradients_naive(Neural& grad, ntype eps, const Matrix<ntype>& tin, const Matrix<ntype>& tout) {
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
    void gradients(Neural& grad, const Matrix<ntype>& tin, const Matrix<ntype>& tout) {
        const size_t p = tin.n;
        for (size_t i = 0; i < n; i++) {
            grad.weight[i].fill(0);
            grad.bias[i].fill(0);
        }
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
                for (size_t j = 0; j < d[l].m; j++)
                    d[l][0][j] *= act.derivative(z[l][0][j]);
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
    void apply_gradients(Neural& grad, ntype rate) {
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
    constexpr void from_stream(std::istream& is) {
        is >> n;
        weight.clear();
        bias.clear();
        activation.clear();

        weight.resize(n);
        bias.resize(n);
        activation.resize(n+1);


        for (size_t i = 0; i < n; i++) {
            size_t rows, cols;
            is >> rows >> cols;
            weight[i] = Matrix<ntype>(rows, cols);
            is >> weight[i];
        }
        for (size_t i = 0; i < n; ++i) {
            size_t rows, cols;
            is >> rows >> cols;
            bias[i] = Matrix<ntype>(rows, cols);
            is >> bias[i];
        }
        activation[0] = Matrix<ntype>(weight[0].m, 1);
        for (size_t i = 1; i <= n; i++) {
            activation[i] = Matrix<ntype>(weight[i-1].n, 1);
        }
    }

    constexpr void to_stream(std::ostream& os,
                        int precision  = 6,
                        int col_width  = 10) const
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

};
template<typename ntype = double> constexpr std::istream& operator >> (std::istream& st, Neural<ntype>& neural) { neural.from_stream(st); return st; }
template<typename ntype = double> constexpr std::ostream& operator << (std::ostream& st, const Neural<ntype>& neural) { neural.to_stream(st); return st; }