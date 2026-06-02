#pragma once

#include "Model.hpp"

namespace cppml {

class LinearRegression : public Model {
private:
    Matrix weights_;
    double bias_;
    double learning_rate_;
    size_t epochs_;
    std::vector<double> loss_history_;

public:
    LinearRegression(double learning_rate = 1e-5, size_t epochs = 1000);

    // Overridden from Model
    void fit(const Matrix& X, const Matrix& y) override;
    Matrix predict(const Matrix& X) const override;
    void evaluate(const Matrix& X, const Matrix& y) const override;
    void saveModel(const std::string& filepath) const override;
    void loadModel(const std::string& filepath) override;

    // Getters
    const Matrix& getWeights() const { return weights_; }
    double getBias() const { return bias_; }
    const std::vector<double>& getLossHistory() const { return loss_history_; }

    // Setters
    void setParameters(const Matrix& weights, double bias) {
        weights_ = weights;
        bias_ = bias;
    }
};

} // namespace cppml
