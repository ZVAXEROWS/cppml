#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>

namespace cppml {

class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;

public:
    Matrix();
    Matrix(size_t rows, size_t cols, double init_val = 0.0);
    Matrix(size_t rows, size_t cols, const std::vector<double>& data);

    // Shape queries
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }
    const std::vector<double>& data() const { return data_; }

    // Element access
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;

    // Operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const; // Matrix multiplication
    Matrix operator*(double scalar) const;       // Scalar multiplication
    
    // Dot product (if vectors)
    double dot(const Matrix& other) const;

    // Transpose
    Matrix transpose() const;

    // For debugging
    void print() const;
};

} // namespace cppml
