#pragma once

#include "Matrix.hpp"

namespace cppml {

class Metrics {
public:
    static double mse(const Matrix& y_true, const Matrix& y_pred);
    static double rmse(const Matrix& y_true, const Matrix& y_pred);
    static double mae(const Matrix& y_true, const Matrix& y_pred);
    static double r2_score(const Matrix& y_true, const Matrix& y_pred);
};

} // namespace cppml
