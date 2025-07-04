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

using namespace std;

class Neuron {
public:
    vector<float> weights;

    Neuron(int dimensions) {
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

int PickClosestNeuronIndex(const vector<Neuron>& neurons, const vector<float>& input) {
    float minDist = CalculateDistance(neurons[0].weights, input);
    int closestIndex = 0;

    for (int i = 1; i < neurons.size(); ++i) {
        float dist = CalculateDistance(neurons[i].weights, input);
        if (dist < minDist) {
            minDist = dist;
            closestIndex = i;
        }
    }

    return closestIndex;
}

void UpdateWeights(Neuron& neuron, const vector<float>& input, float learningRate) {
    for (int i = 0; i < neuron.weights.size(); ++i) {
        neuron.weights[i] += learningRate * (input[i] - neuron.weights[i]);
    }
}

void DecayLearningRate(float& learningRate, float gamma) {
    learningRate *= gamma;
}
void UpdateWeightsWithNeighborhood(vector<Neuron>& neurons, int closestIndex, const vector<float>& input, float learningRate, float sigma) {
    for (int i = 0; i < neurons.size(); ++i) {
        float distance = static_cast<float>(abs(i - closestIndex));
        if (distance <= sigma) {
            float neighborhoodFactor = exp(-distance * distance / (2 * sigma * sigma));
            for (int j = 0; j < neurons[i].weights.size(); ++j) {
                neurons[i].weights[j] += learningRate * neighborhoodFactor * (input[j] - neurons[i].weights[j]);
            }
        }
    }
}

void TrainKohonenNetwork(vector<Neuron>& neurons, const vector<vector<float>>& inputs, int epochs, float initialLearningRate, float gamma) {
    float learningRate = initialLearningRate;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& input : inputs) {
            int closestIndex = PickClosestNeuronIndex(neurons, input);
            UpdateWeights(neurons[closestIndex], input, learningRate);
            float sigma = neurons.size() / 2.0f;
            UpdateWeightsWithNeighborhood(neurons, closestIndex, input, learningRate, sigma);
        }
        DecayLearningRate(learningRate, gamma);
    }
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

    int numNeurons = 3;
    int epochs = 100;
    float initialLearningRate = 0.5f;
    float gamma = 0.95f;

    vector<Neuron> neurons;
    for (int i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(points[0].size());
    }

    TrainKohonenNetwork(neurons, points, epochs, initialLearningRate, gamma);

    // Assign each point to the closest neuron
    vector<int> pointClusterAssignments(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        pointClusterAssignments[i] = PickClosestNeuronIndex(neurons, points[i]);
    }

    // Create output CSV
    ofstream output("Kohonen_result.csv");
    output << "SepalLength,SepalWidth,PetalLength,PetalWidth,TrueClass,Cluster\n";

    for (size_t i = 0; i < labeledPoints.size(); ++i) {
        const vector<float>& features = labeledPoints[i].first;
        const string& label = labeledPoints[i].second;
        for (float val : features) {
            output << val << ",";
        }
        output << label << "," << pointClusterAssignments[i] << "\n";
    }

    output.close();
    cout << "Arquivo 'Kohonen_result.csv' gerado com sucesso." << endl;

    return 0;
}
