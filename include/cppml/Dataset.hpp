#pragma once

#include "Matrix.hpp"
#include <string>

namespace cppml {

class Dataset {
private:
    Matrix X_;
    Matrix y_;

public:
    Dataset();

    // Loads a CSV file, assuming the last column is the target `y`
    // and all preceding columns are features `X`.
    // It will attempt to detect and skip a header row.
    bool loadCSV(const std::string& filepath);

    const Matrix& getX() const { return X_; }
    const Matrix& getY() const { return y_; }
};

} // namespace cppml
