#include "../include/cppml/Metrics.hpp"
#include <cmath>
#include <stdexcept>

namespace cppml {

double Metrics::mse(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols() || y_true.cols() != 1) {
        throw std::invalid_argument("MSE requires matching Nx1 vectors.");
    }
    double sum_error = 0.0;
    size_t n = y_true.rows();
    for (size_t i = 0; i < n; ++i) {
        double err = y_true(i, 0) - y_pred(i, 0);
        sum_error += err * err;
    }
    return sum_error / n;
}

double Metrics::rmse(const Matrix& y_true, const Matrix& y_pred) {
    return std::sqrt(mse(y_true, y_pred));
}

double Metrics::mae(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols() || y_true.cols() != 1) {
        throw std::invalid_argument("MAE requires matching Nx1 vectors.");
    }
    double sum_error = 0.0;
    size_t n = y_true.rows();
    for (size_t i = 0; i < n; ++i) {
        sum_error += std::abs(y_true(i, 0) - y_pred(i, 0));
    }
    return sum_error / n;
}

double Metrics::r2_score(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.rows() != y_pred.rows() || y_true.cols() != y_pred.cols() || y_true.cols() != 1) {
        throw std::invalid_argument("R2 Score requires matching Nx1 vectors.");
    }
    size_t n = y_true.rows();
    
    // Calculate mean of y_true
    double mean_y = 0.0;
    for(size_t i = 0; i < n; ++i) {
        mean_y += y_true(i, 0);
    }
    mean_y /= n;

    double ss_tot = 0.0; // Total sum of squares
    double ss_res = 0.0; // Residual sum of squares

    for(size_t i = 0; i < n; ++i) {
        double y_t = y_true(i, 0);
        double y_p = y_pred(i, 0);
        
        ss_tot += (y_t - mean_y) * (y_t - mean_y);
        ss_res += (y_t - y_p) * (y_t - y_p);
    }

    if (ss_tot == 0.0) return 1.0; // Perfect prediction if variance is 0 and predictions match
    return 1.0 - (ss_res / ss_tot);
}

} // namespace cppml
