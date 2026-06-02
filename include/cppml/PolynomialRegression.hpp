#pragma once

#include "Model.hpp"
#include "LinearRegression.hpp"
#include "PolynomialFeatures.hpp"

namespace cppml {

class PolynomialRegression : public Model {
private:
    int degree_;
    PolynomialFeatures poly_;
    LinearRegression lr_;

public:
    PolynomialRegression(int degree = 2, double learning_rate = 1e-5, size_t epochs = 1000);

    // Overridden from Model
    void fit(const Matrix& X, const Matrix& y) override;
    Matrix predict(const Matrix& X) const override;
    void evaluate(const Matrix& X, const Matrix& y) const override;
    void saveModel(const std::string& filepath) const override;
    void loadModel(const std::string& filepath) override;

    int getDegree() const { return degree_; }
    const LinearRegression& getLinearRegression() const { return lr_; }
};

} // namespace cppml
