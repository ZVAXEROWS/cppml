#pragma once

#include "Matrix.hpp"
#include <string>
#include <vector>

namespace cppml {

class Dataset {
private:
    Matrix X_;
    Matrix y_;
    std::vector<double> feature_means;
    std::vector<double> feature_stds;

public:
    Dataset();

    // Loads a CSV file, assuming the last column is the target `y`
    // and all preceding columns are features `X`.
    // It will attempt to detect and skip a header row.
    bool loadCSV(const std::string& filepath);

	void cleanAndScale();

    const Matrix& getX() const { return X_; }
    const Matrix& getY() const { return y_; }
	const std::vector<double>& getMeans() const { return feature_means; }
	const std::vector<double>& getStds() const { return feature_stds; }
};

} // namespace cppml
