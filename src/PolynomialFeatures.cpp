#include "../include/cppml/PolynomialFeatures.hpp"
#include <cmath>

namespace cppml {

PolynomialFeatures::PolynomialFeatures(int degree) : degree_(degree) {
    if (degree_ < 1) {
        degree_ = 1;
    }
}

Matrix PolynomialFeatures::transform(const Matrix& X) const {
    if (degree_ == 1) {
        return X;
    }

    size_t num_samples = X.rows();
    size_t num_original_features = X.cols();
    size_t num_poly_features = num_original_features * degree_;

    Matrix X_poly(num_samples, num_poly_features);

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_original_features; ++j) {
            double val = X(i, j);
            for (int d = 1; d <= degree_; ++d) {
                // The new feature index
                size_t col_idx = j * degree_ + (d - 1);
                X_poly(i, col_idx) = std::pow(val, d);
            }
        }
    }

    return X_poly;
}

} // namespace cppml
