#include "../include/cppml/Matrix.hpp"

namespace cppml {

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(size_t rows, size_t cols, double init_val)
    : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix dimensions");
    }
}

double& Matrix::operator()(size_t i, size_t j) {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix subscript out of bounds");
    }
    return data_[i * cols_ + j];
}

const double& Matrix::operator()(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix subscript out of bounds");
    }
    return data_[i * cols_ + j];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication");
    }
    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

double Matrix::dot(const Matrix& other) const {
    if (cols_ != 1 || other.cols_ != 1 || rows_ != other.rows_) {
        // Technically dot product is for vectors. 
        // We'll support flattening but typically it's between two Nx1 matrices.
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument("Vector dimensions must match for dot product");
        }
    }
    double sum = 0.0;
    for (size_t i = 0; i < data_.size(); ++i) {
        sum += data_[i] * other.data_[i];
    }
    return sum;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << "\n";
    }
}

} // namespace cppml
