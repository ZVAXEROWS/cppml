#include "../include/cppml/PolynomialRegression.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace cppml {

PolynomialRegression::PolynomialRegression(int degree, double learning_rate, size_t epochs)
    : degree_(degree), poly_(degree), lr_(learning_rate, epochs) {}

void PolynomialRegression::fit(const Matrix& X, const Matrix& y) {
    Matrix X_poly = poly_.transform(X);
    lr_.fit(X_poly, y);
}

Matrix PolynomialRegression::predict(const Matrix& X) const {
    Matrix X_poly = poly_.transform(X);
    return lr_.predict(X_poly);
}

void PolynomialRegression::evaluate(const Matrix& X, const Matrix& y) const {
    Matrix X_poly = poly_.transform(X);
    lr_.evaluate(X_poly, y);
}


void PolynomialRegression::saveModel(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open file for saving model.");
    }

    // Write degree
    out.write(reinterpret_cast<const char*>(&degree_), sizeof(degree_));

    // Get LR params
    double bias = lr_.getBias();
    out.write(reinterpret_cast<const char*>(&bias), sizeof(bias));

    const Matrix& weights = lr_.getWeights();
    size_t w_rows = weights.rows();
    out.write(reinterpret_cast<const char*>(&w_rows), sizeof(w_rows));

    out.write(reinterpret_cast<const char*>(weights.data().data()), w_rows * sizeof(double));

    out.close();
    std::cout << "Polynomial Model saved to " << filepath << std::endl;
}

void PolynomialRegression::loadModel(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open file for loading model.");
    }

    // Read degree
    in.read(reinterpret_cast<char*>(&degree_), sizeof(degree_));
    poly_ = PolynomialFeatures(degree_);

    // Read LR params
    double bias;
    in.read(reinterpret_cast<char*>(&bias), sizeof(bias));

    size_t w_rows;
    in.read(reinterpret_cast<char*>(&w_rows), sizeof(w_rows));

    std::vector<double> w_data(w_rows);
    in.read(reinterpret_cast<char*>(w_data.data()), w_rows * sizeof(double));

    in.close();
    
    Matrix weights(w_rows, 1, w_data);
    lr_.setParameters(weights, bias);
    std::cout << "Polynomial Model loaded from " << filepath << std::endl;
}

} // namespace cppml
