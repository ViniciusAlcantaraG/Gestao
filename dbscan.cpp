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

    // Executa DBSCAN, que retorna a estrutura original. Nenhuma mudança aqui.
    map<int, vector<vector<float>>> clusters = DBSCAN(points, 5, 0.5);

    // Cria o CSV de saída
    ofstream output("DBSCAN_result.csv");
    output << "SepalLength,SepalWidth,PetalLength,PetalWidth,TrueClass,Cluster\n";

    // Para cada ponto original...
    for (const auto& entry : labeledPoints) {
        const vector<float>& features = entry.first;
        const string& label = entry.second;
        int foundClusterId = -1; // -1 (ruído) é o padrão se não for encontrado em nenhum cluster

        // ...procure em qual cluster ele está.
        // Isso é uma busca reversa.
        for (const auto& pair : clusters) {
            int currentClusterId = pair.first;
            const vector<vector<float>>& pointsInCluster = pair.second;

            // Procura o ponto 'features' na lista de pontos do cluster atual
            if (find(pointsInCluster.begin(), pointsInCluster.end(), features) != pointsInCluster.end()) {
                foundClusterId = currentClusterId;
                break; // Encontrou, pode parar de procurar nos outros clusters
            }
        }

        // Escreve os dados no CSV
        for (float val : features) {
            output << val << ",";
        }
        output << label << "," << foundClusterId << "\n";
    }

    output.close();
    // Corrigindo também a mensagem de saída
    cout << "Arquivo 'DBSCAN_result.csv' gerado com sucesso." << endl;

    return 0;
}
