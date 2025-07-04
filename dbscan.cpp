#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include <set>
#include <map>
#include <iostream>
#include <ostream>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace std;

// NENHUMA ALTERAÇÃO NESTA FUNÇÃO
vector<vector<string>> ReadCsv(const string& filename){
    vector<vector<string>> data;
    ifstream file(filename);
    if (!file.is_open()){
        cerr << "Impossível abir arquivo: " << filename << endl;
        return data;
    }
    string line;
    while (getline(file, line)){
        vector<string> row;
        stringstream ss(line);
        string cell;
        while(getline(ss, cell, ',')){
            row.push_back(cell);
        }
        data.push_back(row);
    }
    file.close();
    return data;
}

// NENHUMA ALTERAÇÃO NESTA FUNÇÃO
float CalculateDistance(vector<float> a, vector<float> b){
    float dist = 0;
    for (int i = 0; i < a.size(); ++i){
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    float euclideanDistance = sqrt(dist);
    return euclideanDistance;
}

// NENHUMA ALTERAÇÃO NESTA FUNÇÃO
bool IsInRadius(vector<float> center, vector<float> point, float radius){
    float distance = CalculateDistance(center, point);
    if (distance <= radius){
        return true;
    }
    else{
        return false; 
    }
}

// NENHUMA ALTERAÇÃO NESTA FUNÇÃO
vector<vector<float>> FindNeighbourhood(vector<float> center, vector<vector<float>> points, float radius){
    vector<vector<float>> neighbourhood;
    for (int i = 0; i < points.size(); ++i){
        if (IsInRadius(center, points[i], radius)){
            neighbourhood.push_back(points[i]);
        }
    }
    return neighbourhood;
}

// NENHUMA ALTERAÇÃO NESTA FUNÇÃO
map<int, vector<vector<float>>> DBSCAN(vector<vector<float>> points, int minPoints, float radius) {
    int n = points.size();
    vector<int> labels(n, 0); // 0: unvisited, -1: noise, >0: cluster id
    int clusterId = 0;
    for (int i = 0; i < n; ++i) {
        if (labels[i] != 0) continue;
        vector<int> neighbors;
        for (int j = 0; j < n; ++j) {
            if (IsInRadius(points[i], points[j], radius))
                neighbors.push_back(j);
        }
        if (neighbors.size() < minPoints) {
            labels[i] = -1;
            continue;
        }
        ++clusterId;
        labels[i] = clusterId;
        size_t idx = 0;
        while (idx < neighbors.size()) {
            int neighborIdx = neighbors[idx];
            if (labels[neighborIdx] == -1)
                labels[neighborIdx] = clusterId;
            if (labels[neighborIdx] != 0) {
                ++idx;
                continue;
            }
            labels[neighborIdx] = clusterId;
            vector<int> neighborNeighbors;
            for (int k = 0; k < n; ++k) {
                if (IsInRadius(points[neighborIdx], points[k], radius))
                    neighborNeighbors.push_back(k);
            }
            if (neighborNeighbors.size() >= minPoints) {
                neighbors.insert(neighbors.end(), neighborNeighbors.begin(), neighborNeighbors.end());
            }
            ++idx;
        }
    }
    map<int, vector<vector<float>>> clusters;
    for (int i = 0; i < n; ++i) {
        if (labels[i] > 0)
            clusters[labels[i]].push_back(points[i]);
    }
    return clusters;
}

int main() {

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

    vector<float> radiusList = {0.3, 0.5, 0.7, 1.0};
    vector<int> minPointsList = {3, 5, 7, 10};

    // Cria o CSV de saída para resultados
    ofstream output("DBSCAN_result.csv");
    output << "SepalLength,SepalWidth,PetalLength,PetalWidth,TrueClass,Cluster,Radius,MinPoints\n";

    // Cria o CSV de saída para tempos
    ofstream timeOutput("DBSCAN_times.csv");
    timeOutput << "Radius,MinPoints,ExecutionTimeMs\n";

    for (float radius : radiusList) {
        for (int minPoints : minPointsList) {
            auto start = chrono::high_resolution_clock::now();

            map<int, vector<vector<float>>> clusters = DBSCAN(points, minPoints, radius);

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

            // Escreve tempo no CSV de tempos
            timeOutput << radius << "," << minPoints << "," << duration << "\n";

            // Para cada ponto original...
            for (const auto& entry : labeledPoints) {
                const vector<float>& features = entry.first;
                const string& label = entry.second;
                int foundClusterId = -1; // -1 (ruído) é o padrão se não for encontrado em nenhum cluster

                // ...procure em qual cluster ele está.
                for (const auto& pair : clusters) {
                    int currentClusterId = pair.first;
                    const vector<vector<float>>& pointsInCluster = pair.second;

                    if (find(pointsInCluster.begin(), pointsInCluster.end(), features) != pointsInCluster.end()) {
                        foundClusterId = currentClusterId;
                        break;
                    }
                }

                // Escreve os dados no CSV
                for (float val : features) {
                    output << val << ",";
                }
                output << label << "," << foundClusterId << "," << radius << "," << minPoints << "\n";
            }
        }
    }

    output.close();
    timeOutput.close();
    cout << "Arquivo 'DBSCAN_result.csv' e 'DBSCAN_times.csv' gerados com sucesso." << endl;

    return 0;
}
