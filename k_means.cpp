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

float CalculateDistance(vector<float> a, vector<float> b) {
    float dist = 0;
    for (int i = 0; i < a.size(); ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

map<int, vector<vector<float>>> AssignToCentroid(
    vector<vector<float>>& centroids,
    map<int, vector<vector<float>>>& clusters,
    vector<vector<float>>& points) {

    clusters.clear();
    for (int j = 0; j < points.size(); j++) {
        float minDist = CalculateDistance(points[j], centroids[0]);
        int minIdx = 0;
        for (int i = 1; i < centroids.size(); i++) {
            float dist = CalculateDistance(points[j], centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                minIdx = i;
            }
        }
        clusters[minIdx].push_back(points[j]);
    }
    return clusters;
}

vector<vector<float>> PickRandomCentroids(int k, vector<vector<float>> points) {
    vector<vector<float>> centroids;
    random_device rd;
    mt19937 gen(rd());
    set<int> usedIdx;
    while (centroids.size() < k) {
        uniform_int_distribution<size_t> dist(0, points.size() - 1);
        int randomIdx = dist(gen);
        if (usedIdx.find(randomIdx) == usedIdx.end()) {
            centroids.push_back(points[randomIdx]);
            usedIdx.emplace(randomIdx);
        }
    }
    return centroids;
}

vector<vector<float>> CalculateNewCentroids(map<int, vector<vector<float>>>& clusters, vector<vector<float>> centroids) {
    for (int i = 0; i < centroids.size(); ++i) {
        if (clusters[i].empty()) continue;
        vector<float> means(centroids[0].size(), 0.0f);
        for (auto& point : clusters[i]) {
            for (int k = 0; k < point.size(); ++k) {
                means[k] += point[k];
            }
        }
        for (int k = 0; k < means.size(); ++k) {
            means[k] /= clusters[i].size();
        }
        centroids[i] = means;
    }
    return centroids;
}

map<int, vector<vector<float>>> KMeans(int k, vector<vector<float>> points) {
    bool convergence = false;
    vector<vector<float>> centroids = PickRandomCentroids(k, points);
    map<int, vector<vector<float>>> clusters;

    while (!convergence) {
        clusters = AssignToCentroid(centroids, clusters, points);
        vector<vector<float>> newCentroids = CalculateNewCentroids(clusters, centroids);
        if (newCentroids == centroids) {
            convergence = true;
        } else {
            centroids = newCentroids;
        }
    }
    return clusters;
}

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

int main() {
    int numberOfClusters = 3;

    vector<vector<string>> data = ReadCsv("Iris.csv");
    vector<vector<float>> points;
    vector<pair<vector<float>, string>> labeledPoints;

    for (size_t i = 1; i < data.size(); ++i) {
        vector<float> point;
        for (size_t j = 1; j < data[i].size() - 1; ++j) {
            point.push_back(stof(data[i][j]));
        }
        string label = data[i].back();
        points.push_back(point);
        labeledPoints.emplace_back(point, label);
    }

    // Executa KMeans
    map<int, vector<vector<float>>> clusters = KMeans(numberOfClusters, points);

    // Associa cada ponto ao cluster a que pertence
    map<vector<float>, int> pointToCluster;
    for (const auto& pair : clusters) {
        int clusterId = pair.first;
        for (const auto& point : pair.second) {
            pointToCluster[point] = clusterId;
        }
    }

    // Cria o CSV de saída
    ofstream output("K_means_result.csv");
    output << "SepalLength,SepalWidth,PetalLength,PetalWidth,TrueClass,Cluster\n";

    for (const auto& entry : labeledPoints) {
        const vector<float>& features = entry.first;
        const string& label = entry.second;
        for (float val : features) {
            output << val << ",";
        }
        output << label << "," << pointToCluster[features] << "\n";
    }

    output.close();
    cout << "Arquivo 'K_means_result.csv' gerado com sucesso." << endl;

    return 0;
}
