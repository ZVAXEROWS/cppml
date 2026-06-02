#include "../include/cppml/LinearRegression.hpp"
#include "../include/cppml/Metrics.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace cppml {

LinearRegression::LinearRegression(double learning_rate, size_t epochs)
    : bias_(0.0), learning_rate_(learning_rate), epochs_(epochs) {}

void LinearRegression::fit(const Matrix& X, const Matrix& y) {
    if (X.rows() != y.rows() || y.cols() != 1) {
        throw std::invalid_argument("Features and target size mismatch, or target is not a vector.");
    }

    size_t num_samples = X.rows();
    size_t num_features = X.cols();

    // Initialize weights to zero
    weights_ = Matrix(num_features, 1, 0.0);
    bias_ = 0.0;
    loss_history_.clear();

    for (size_t epoch = 0; epoch < epochs_; ++epoch) {
        // Predictions: y_pred = X * W + b
        Matrix y_pred = predict(X);

        // Error: y_pred - y
        Matrix error = y_pred - y;

        // Gradients
        // dW = (1/N) * X^T * error
        Matrix dW = X.transpose() * error * (1.0 / num_samples);
        
        // db = (1/N) * sum(error)
        double db = 0.0;
        for (size_t i = 0; i < error.rows(); ++i) {
            db += error(i, 0);
        }
        db /= num_samples;

        // Update parameters
        weights_ = weights_ - (dW * learning_rate_);
        bias_ -= learning_rate_ * db;

        // Calculate and log loss
        double current_loss = Metrics::mse(y, y_pred);
        loss_history_.push_back(current_loss);

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << current_loss << std::endl;
        }
    }
}

Matrix LinearRegression::predict(const Matrix& X) const {
    if (weights_.size() == 0 || X.cols() != weights_.rows()) {
        throw std::invalid_argument("Model not initialized or feature dimension mismatch in prediction.");
    }

    Matrix y_pred = X * weights_;
    // Add bias
    for (size_t i = 0; i < y_pred.rows(); ++i) {
        y_pred(i, 0) += bias_;
    }
    return y_pred;
}

void LinearRegression::evaluate(const Matrix& X, const Matrix& y) const {
    Matrix y_pred = predict(X);
    double mse = Metrics::mse(y, y_pred);
    double rmse = Metrics::rmse(y, y_pred);
    double mae = Metrics::mae(y, y_pred);
    double r2 = Metrics::r2_score(y, y_pred);

    std::cout << "Evaluation Metrics:" << std::endl;
    std::cout << "MSE:   " << mse << std::endl;
    std::cout << "RMSE:  " << rmse << std::endl;
    std::cout << "MAE:   " << mae << std::endl;
    std::cout << "R2:    " << r2 << std::endl;
}

void LinearRegression::saveModel(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open file for saving model.");
    }

    // Write bias
    out.write(reinterpret_cast<const char*>(&bias_), sizeof(bias_));

    // Write weights dimension
    size_t w_rows = weights_.rows();
    out.write(reinterpret_cast<const char*>(&w_rows), sizeof(w_rows));

    // Write weights data
    out.write(reinterpret_cast<const char*>(weights_.data().data()), w_rows * sizeof(double));

    out.close();
    std::cout << "Model saved to " << filepath << std::endl;
}

void LinearRegression::loadModel(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open file for loading model.");
    }

    // Read bias
    in.read(reinterpret_cast<char*>(&bias_), sizeof(bias_));

    // Read weights dimension
    size_t w_rows;
    in.read(reinterpret_cast<char*>(&w_rows), sizeof(w_rows));

    // Read weights data
    std::vector<double> w_data(w_rows);
    in.read(reinterpret_cast<char*>(w_data.data()), w_rows * sizeof(double));

    weights_ = Matrix(w_rows, 1, w_data);

    in.close();
    std::cout << "Model loaded from " << filepath << std::endl;
}

} // namespace cppml
