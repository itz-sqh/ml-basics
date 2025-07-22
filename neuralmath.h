#pragma once
#include<random>
#include<vector>
#include<iomanip>
#include<cassert>
#include<cmath>
#include<algorithm>

template<typename mtype = float>
mtype rand_val() {
    return static_cast<mtype>(rand())/static_cast<mtype>(RAND_MAX);
}

template<typename mtype = float>
struct Matrix {

    size_t n,m;
    std::vector<std::vector<mtype>> a;

    constexpr Matrix();
    constexpr Matrix(size_t n, size_t m, bool identity = false);
    constexpr Matrix(std::initializer_list<std::initializer_list<mtype>> init);

    Matrix operator+=(const Matrix& other);
    Matrix operator+(const Matrix& other) const;

    Matrix operator-=(const Matrix& other);
    Matrix operator-(const Matrix& other) const;

    Matrix operator*=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;

    Matrix operator*(const std::vector<mtype>& vec) const;
    Matrix operator*=(const std::vector<mtype>& vec);

    Matrix operator*=(mtype val);
    Matrix operator*(mtype val) const;


    constexpr std::vector<mtype>& operator[] (size_t i);
    constexpr const std::vector<mtype>& operator[] (size_t i) const;

    constexpr bool operator==(const Matrix& other) const;
    constexpr bool operator!=(const Matrix& other) const;

    constexpr mtype& at(size_t i, size_t j);
    constexpr const mtype& at(size_t i, size_t j) const;

    Matrix row(size_t r) const;
    Matrix col(size_t c) const;
    Matrix subMatrix(size_t r1, size_t c1, size_t r2, size_t c2) const;

    Matrix transpose() const;

    constexpr void fill(mtype val);

    template<typename ftype>
    void apply(ftype func);

    constexpr void rand_init(mtype min = {0}, mtype max = {1});

    constexpr void to_stream(
        std::ostream& os,
        int precision = 4,
        int col_width = 4,
        int offset = 0) const;
};






template<typename mtype>
constexpr Matrix<mtype>::Matrix() : n(0), m(0) {}


template<typename mtype>
constexpr Matrix<mtype>::Matrix(size_t n, size_t m, const bool identity) : n(n),m(m), a(n,std::vector<mtype>(m,0)) {
    if (identity) {
        assert(!(identity && n!=m) && "Identity Matrix must be a square Matrix");
        for (size_t i = 0; i < n; i++) {
            a[i][i] = 1;
        }
    }
}

template<typename mtype>
constexpr Matrix<mtype>::Matrix(std::initializer_list<std::initializer_list<mtype>> init) : n(init.size()), m(init.begin()->size()), a(n,std::vector<mtype>(m)) {
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

template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator+=(const Matrix &other) {
    assert(n == other.n && m == other.m && "Matrices must be the same size");
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            a[i][j] += other.a[i][j];
        }
    }
    return *this;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator+(const Matrix& other) const {
    return Matrix(*this) += other;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator-=(const Matrix& other) {
    assert(n == other.n && m == other.m && "Matrices must be the same size");
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            a[i][j] -= other.a[i][j];
        }
    }
    return *this;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator-(const Matrix& other) const {
    return Matrix(*this) -= other;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator*=(const Matrix& other) {
    *this = *this * other;
    return *this;
}
template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator*(const Matrix& other) const {
    assert(m == other.n && "Inner sizes must be equal");
    Matrix res(n,other.m);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < other.m; j++)
            for (size_t k = 0; k < m; k++)
                res[i][j] += a[i][k] * other.a[k][j];
    return res;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator*(const std::vector<mtype>& vec) const {
    assert(m == vec.size() && "Matrix columns must match vector size");

    Matrix res(n,1);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            res[i][0] += a[i][j] * vec[j];
    return res;
}
template<typename mtype>

Matrix<mtype> Matrix<mtype>::operator*=(const std::vector<mtype>& vec) {
    *this = *this * vec;
    return *this;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator*(mtype val) const {
    Matrix res(n,m);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            res[i][j] = a[i][j] * val;
    return res;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::operator*=(mtype val) {
    *this = *this * val;
    return *this;
}

template<typename mtype>
constexpr std::vector<mtype>& Matrix<mtype>::operator[] (size_t i) {
    assert(i < n && "Index out of bounds");
    return a[i];
}

template<typename mtype>
constexpr const std::vector<mtype>& Matrix<mtype>::operator[] (size_t i) const {
    assert(i < n && "Index out of bounds");
    return a[i];
}

template<typename mtype>
constexpr bool Matrix<mtype>::operator==(const Matrix& other) const {
    assert(n == other.n && m == other.m && "Matrices must have same dimensions");
    return a == other.a;
}

template<typename mtype>
constexpr bool Matrix<mtype>::operator!=(const Matrix& other) const {
    assert(n == other.n && m == other.m && "Matrices must have same dimensions");
    return a != other.a;
}

template<typename mtype>
constexpr mtype& Matrix<mtype>::at(size_t i, size_t j) {
    assert(i<n && j < m && "Index out of bounds");
    return a[i][j];
}

template<typename mtype>
constexpr const mtype& Matrix<mtype>::at(size_t i, size_t j) const {
    assert(i<n && j < m && "Index out of bounds");
    return a[i][j];
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::row(size_t r) const {
    assert(r < n && "Row index out of bounds");
    Matrix res(1,m);
    for (size_t i = 0; i < m; i++)
        res[0][i] = a[r][i];
    return res;
}


template<typename mtype>
Matrix<mtype> Matrix<mtype>::col(size_t c) const {
    assert(c < m && "Column index out of bounds");
    Matrix res(n, 1);
    for (size_t i = 0; i < n; ++i)
        res[i][0] = a[i][c];
    return res;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::subMatrix(const size_t r1, const size_t c1, const size_t r2, const size_t c2) const {
    assert(r2 < n && c2 < m && r1 <= r2 && c1 <= c2);
    Matrix res(r2 - r1 + 1, c2 - c1 + 1);
    for (size_t i = r1; i <= r2; ++i)
        for (size_t j = c1; j <= c2; ++j)
            res.a[i - r1][j - c1] = a[i][j];
    return res;
}

template<typename mtype>
Matrix<mtype> Matrix<mtype>::transpose() const {
    Matrix res(m,n);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            res.a[j][i] = a[i][j];
    return res;
}

template<typename mtype>
constexpr void Matrix<mtype>::fill(mtype val) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            a[i][j] = val;
}

template<typename mtype>
template<typename ftype>
void Matrix<mtype>::apply(ftype func) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            a[i][j] = func(a[i][j]);
}

template<typename mtype>
constexpr void Matrix<mtype>::rand_init(mtype min, mtype max) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            a[i][j] = rand_val<mtype>()*(max-min)+min;
}


template<typename mtype>
constexpr void Matrix<mtype>::to_stream(
    std::ostream& os,
    int precision,
    int col_width,
    int offset) const
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

template<typename mtype> constexpr std::ostream& operator << (std::ostream& st, const Matrix<mtype>& m) { m.to_stream(st); return st; }



template<typename atype>
struct Sigmoid {
    atype operator()(atype x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }
    atype derivative(atype x) const {
        atype s = (*this)(x);
        return s * (1 - s);
    }
};

template<typename atype>
struct ReLU {
    atype operator()(atype x) const {
        return (x > 0) ? x : 0;
    }
    atype derivative(atype x) const {
        return (x > 0) ? 1 : 0;
    }
};

template<typename atype>
struct Gaussian {
    atype operator()(const atype x) const {
        return std::exp(-x * x);
    }
    atype derivative(const atype x) const {
        return -2 * x * (*this)(x);
    }
};

