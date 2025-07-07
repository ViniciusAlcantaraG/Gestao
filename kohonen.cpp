#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include <set>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>


using namespace std;

class Neuron {
public:
    int row, col;
    vector<float> weights;

    Neuron(int dimensions, int row_, int col_) : row(row_), col(col_) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(-1.0, 1.0);
        for (int i = 0; i < dimensions; ++i) {
            weights.push_back(dist(gen));
        }
    }
};

vector<vector<string>> ReadCsv(const string& filename) {
    vector<vector<string>> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Impossível abrir arquivo: " << filename << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        vector<string> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

float CalculateDistance(const vector<float>& a, const vector<float>& b) {
    float dist = 0;
    for (int i = 0; i < a.size(); ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

// Find BMU (Best Matching Unit) in the 2D grid
pair<int, int> PickBMU(const vector<vector<Neuron>>& network, const vector<float>& input) {
    float minDist = -1;
    pair<int, int> bmu = {0, 0};
    for (int i = 0; i < network.size(); ++i) {
        for (int j = 0; j < network[i].size(); ++j) {
            float dist = 0;
            for (int k = 0; k < input.size(); ++k) {
                dist += pow(input[k] - network[i][j].weights[k], 2);
            }
            dist = sqrt(dist);
            if (minDist < 0 || dist < minDist) {
                minDist = dist;
                bmu = {i, j};
            }
        }
    }
    return bmu;
}

// Update weights for all neurons in the 2D grid using neighborhood function
void UpdateWeightsWithNeighborhood2D(vector<vector<Neuron>>& network, pair<int, int> bmu, float eta, float sigma, const vector<float>& input) {
    int rows = network.size();
    int cols = network[0].size();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float r = sqrt(pow(bmu.first - i, 2) + pow(bmu.second - j, 2));
            float neighborhood = exp(- (r * r) / (2 * sigma * sigma));
            for (int k = 0; k < input.size(); ++k) {
                network[i][j].weights[k] += neighborhood * (input[k] - network[i][j].weights[k]) * eta;
            }
        }
    }
}

// Linear decay for learning rate
void LinearDecayLearningRate(float& learningRate, int k, int maxIterations) {
    learningRate *= (1.0f - (float(k) / float(maxIterations)));
}

// Linear decay for sigma
float LinearSigma(int k, int maxIterations, int rows, int cols) {
    float initialSigma = max(rows, cols) / 2;
    return initialSigma * (1 - (float(k) / float(maxIterations)));
}

// Train SOM using 2D grid
void TrainKohonenNetwork2D(vector<vector<Neuron>>& network, const vector<vector<float>>& inputs, int epochs, float initialLearningRate) {
    int rows = network.size();
    int cols = network[0].size();
    float learningRate = initialLearningRate;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& input : inputs) {
            auto bmu = PickBMU(network, input);
            float sigma = LinearSigma(epoch, epochs, rows, cols);
            UpdateWeightsWithNeighborhood2D(network, bmu, learningRate, sigma, input);
        }
        LinearDecayLearningRate(learningRate, epoch, epochs);
    }
}

// Assign each point to the closest neuron (returns cluster id as integer)
int GetClusterId(int row, int col, int cols) {
    return row * cols + col;
}

int main() {
    vector<vector<string>> data = ReadCsv("Iris.csv");
    vector<vector<float>> points;
    vector<pair<vector<float>, string>> labeledPoints;

    // Skip header row
    for (size_t i = 1; i < data.size(); ++i) {
        vector<float> point;
        for (size_t j = 1; j < data[i].size() - 1; ++j) {
            point.push_back(stof(data[i][j]));
        }
        string label = data[i].back();
        points.push_back(point);
        labeledPoints.emplace_back(point, label);
    }

    // Parameter ranges
    vector<int> rowOptions = {2, 5};
    vector<int> colOptions = {2, 5};
    vector<float> learningRates = {0.1, 0.2, 0.5};
    int epochs = 100;

    // Prepare output files
    ofstream output("Kohonen_result.csv");
    output << "Rows,Cols,LearningRate,SepalLength,SepalWidth,PetalLength,PetalWidth,TrueClass,Cluster\n";
    ofstream timing("Kohonen_times.csv");
    timing << "Rows,Cols,LearningRate,TimeMs\n";

    for (int rows : rowOptions) {
        for (int cols : colOptions) {
            for (float initialLearningRate : learningRates) {
                auto start = chrono::high_resolution_clock::now();

                // Initialize 2D grid of neurons
                vector<vector<Neuron>> network(rows, vector<Neuron>(cols, Neuron(points[0].size(), 0, 0)));
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        network[i][j] = Neuron(points[0].size(), i, j);
                    }
                }

                TrainKohonenNetwork2D(network, points, epochs, initialLearningRate);

                // Assign each point to the closest neuron (BMU)
                vector<int> pointClusterAssignments(points.size());
                for (size_t i = 0; i < points.size(); ++i) {
                    auto bmu = PickBMU(network, points[i]);
                    pointClusterAssignments[i] = GetClusterId(bmu.first, bmu.second, cols);
                }

                // Write results to CSV
                for (size_t i = 0; i < labeledPoints.size(); ++i) {
                    const vector<float>& features = labeledPoints[i].first;
                    const string& label = labeledPoints[i].second;
                    output << rows << "," << cols << "," << initialLearningRate << ",";
                    for (float val : features) {
                        output << val << ",";
                    }
                    output << label << "," << pointClusterAssignments[i] << "\n";
                }

                auto end = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
                timing << rows << "," << cols << "," << initialLearningRate << "," << duration << "\n";
            }
        }
    }

    output.close();
    timing.close();
    cout << "Arquivo 'Kohonen_result.csv' e 'Kohonen_times.csv' gerados com sucesso." << endl;

    return 0;
}
