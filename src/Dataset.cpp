#include "../include/cppml/Dataset.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>

namespace cppml {

Dataset::Dataset() {}

void Dataset::cleanAndScale() {
    int rows = X_.rows();
    int cols = X_.cols();

    feature_means.resize(cols, 0.0);
    feature_stds.resize(cols, 0.0);

    for (int j = 0; j < cols; ++j) {
        // 1. Data Cleansing: Handle missing data / Compute Mean
        double sum = 0.0;
        int valid_count = 0;
        for (int i = 0; i < rows; ++i) {
            // Assuming 0.0 or NaN representation might be present
            if (!std::isnan(X_(i, j))) {
                sum += X_(i, j);
                valid_count++;
            }
        }

        double mean = (valid_count > 0) ? (sum / valid_count) : 0.0;
        feature_means[j] = mean;

        // Clean missing values by replacing them with the column mean
        for (int i = 0; i < rows; ++i) {
            if (std::isnan(X_(i, j))) {
                X_(i, j) = mean;
            }
        }

        // 2. Compute Standard Deviation
        double variance_sum = 0.0;
        for (int i = 0; i < rows; ++i) {
            variance_sum += std::pow(X_(i, j) - mean, 2);
        }
        double std_dev = std::sqrt(variance_sum / rows);
        if (std_dev == 0.0) std_dev = 1.0; // Prevent division by zero
        feature_stds[j] = std_dev;

        // 3. Apply Z-score Normalization
        for (int i = 0; i < rows; ++i) {
            X_(i, j) = (X_(i, j) - mean) / std_dev;
        }
    }
    std::cout << "Data cleansing and Z-score optimization complete.\n";
}

bool Dataset::loadCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::vector<std::vector<double>> features;
    std::vector<double> targets;
    
    bool isFirstRow = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;
        bool isHeader = false;

        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);

            if (isFirstRow) {
                // Check if the token is a number
                try {
                    (void)std::stod(token);
                } catch (const std::invalid_argument&) {
                    isHeader = true;
                    break;
                }
            }
            if (!isHeader) {
                try {
                    row.push_back(std::stod(token));
                } catch (const std::invalid_argument&) {
                    std::cerr << "Error parsing token to double: " << token << std::endl;
                    return false;
                }
            }
        }

        if (isFirstRow) {
            isFirstRow = false;
            if (isHeader) {
                continue; // Skip the header row
            }
        }

        if (!row.empty()) {
            targets.push_back(row.back());
            row.pop_back();
            features.push_back(row);
        }
    }

    if (features.empty()) {
        std::cerr << "Error: No valid data found in CSV." << std::endl;
        return false;
    }

    size_t numRows = features.size();
    size_t numCols = features[0].size();

    // Fill X
    X_ = Matrix(numRows, numCols);
    for (size_t i = 0; i < numRows; ++i) {
        if (features[i].size() != numCols) {
            std::cerr << "Error: Inconsistent number of columns in CSV." << std::endl;
            return false;
        }
        for (size_t j = 0; j < numCols; ++j) {
            X_(i, j) = features[i][j];
        }
    }

    // Fill y
    y_ = Matrix(numRows, 1);
    for (size_t i = 0; i < numRows; ++i) {
        y_(i, 0) = targets[i];
    }

    return true;
}

} // namespace cppml
