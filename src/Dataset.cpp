#include "../include/cppml/Dataset.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

namespace cppml {

Dataset::Dataset() {}

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
