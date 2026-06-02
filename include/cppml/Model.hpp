#pragma once

#include "Matrix.hpp"
#include <string>

namespace cppml {

class Model {
public:
    virtual ~Model() = default;

    // Train the model
    virtual void fit(const Matrix& X, const Matrix& y) = 0;

    // Predict new values
    virtual Matrix predict(const Matrix& X) const = 0;

    // Evaluate the model on testing data (could print or return metrics)
    virtual void evaluate(const Matrix& X, const Matrix& y) const = 0;

    // Model persistence
    virtual void saveModel(const std::string& filepath) const = 0;
    virtual void loadModel(const std::string& filepath) = 0;
};

} // namespace cppml
