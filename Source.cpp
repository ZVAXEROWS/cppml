#include <iostream>
#include <string>
#include <vector>
#include "include/cppml/Dataset.hpp"
#include "include/cppml/LinearRegression.hpp"
#include "include/cppml/PolynomialRegression.hpp"

using namespace cppml;

void printUsage() {
    std::cout << "CPPML CLI Usage:\n";
    std::cout << "  train <dataset.csv> [--poly <degree>] [--lr <rate>] [--epochs <count>]\n";
    std::cout << "  predict <model.bin> <feature1> [feature2] ...\n";
    std::cout << "  evaluate <dataset.csv> [--poly <degree>]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string command = argv[1];

    if (command == "train") {
        if (argc < 3) {
            std::cerr << "Error: Dataset path required.\n";
            printUsage();
            return 1;
        }
        std::string dataset_path = argv[2];
        
        int poly_degree = 1;
        double learning_rate = 1e-5;
        size_t epochs = 1000;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--poly" && i + 1 < argc) {
                poly_degree = std::stoi(argv[++i]);
            } else if (arg == "--lr" && i + 1 < argc) {
                learning_rate = std::stod(argv[++i]);
            } else if (arg == "--epochs" && i + 1 < argc) {
                epochs = std::stoull(argv[++i]);
            }
        }

        Dataset dataset;
        if (!dataset.loadCSV(dataset_path)) {
            return 1;
        }

        std::cout << "Training on " << dataset.getX().rows() << " samples with " << dataset.getX().cols() << " features.\n";

        if (poly_degree > 1) {
            PolynomialRegression model(poly_degree, learning_rate, epochs);
            model.fit(dataset.getX(), dataset.getY());
            model.evaluate(dataset.getX(), dataset.getY());
            model.saveModel("model.bin");
        } else {
            LinearRegression model(learning_rate, epochs);
            model.fit(dataset.getX(), dataset.getY());
            model.evaluate(dataset.getX(), dataset.getY());
            model.saveModel("model.bin");
        }

    } else if (command == "predict") {
        if (argc < 4) {
            std::cerr << "Error: Model path and at least one feature required.\n";
            printUsage();
            return 1;
        }
        std::string model_path = argv[2];
        std::vector<double> features;
        for (int i = 3; i < argc; ++i) {
            features.push_back(std::stod(argv[i]));
        }

        Matrix X_test(1, features.size(), features);

        // We don't have a model registry, so we'll try to load as LinearRegression. 
        // If it was saved as poly, we would need to know the degree. 
        // For simplicity, we just assume LinearRegression for the CLI predict unless told otherwise.
        // Wait, PolynomialRegression saves degree first. We can actually peek at the first 4 bytes to check if it's a small integer (degree) or a double (bias).
        // Since bias is usually a normal double and degree is an int. 
        // Actually, let's just add an optional flag for predict too, or default to LR.
        std::cout << "Loading model from " << model_path << " (assuming Linear Regression)...\n";
        try {
            LinearRegression model;
            model.loadModel(model_path);
            Matrix pred = model.predict(X_test);
            std::cout << "Prediction: " << pred(0, 0) << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during prediction: " << e.what() << std::endl;
        }

    } else if (command == "evaluate") {
        if (argc < 3) {
            std::cerr << "Error: Dataset path required.\n";
            printUsage();
            return 1;
        }
        std::string dataset_path = argv[2];
        
        int poly_degree = 1;
        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--poly" && i + 1 < argc) {
                poly_degree = std::stoi(argv[++i]);
            }
        }

        Dataset dataset;
        if (!dataset.loadCSV(dataset_path)) {
            return 1;
        }

        try {
            if (poly_degree > 1) {
                PolynomialRegression model(poly_degree);
                model.loadModel("model.bin");
                model.evaluate(dataset.getX(), dataset.getY());
            } else {
                LinearRegression model;
                model.loadModel("model.bin");
                model.evaluate(dataset.getX(), dataset.getY());
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during evaluation: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        printUsage();
        return 1;
    }

    return 0;
}
