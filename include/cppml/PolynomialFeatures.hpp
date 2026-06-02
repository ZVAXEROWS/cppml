#pragma once

#include "Matrix.hpp"

namespace cppml {

class PolynomialFeatures {
private:
    int degree_;

public:
    PolynomialFeatures(int degree = 2);

    // Transform input Matrix X to include polynomial features.
    // For each feature x, it adds x^2, x^3, ..., x^degree
    Matrix transform(const Matrix& X) const;

    int getDegree() const { return degree_; }
};

} // namespace cppml
