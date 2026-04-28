#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <chrono>

using namespace std;

class Graph
{
private:
    int vertices;
    vector<vector<int>> adjacencyList;

public:
    Graph(int v) : vertices(v)
    {
        adjacencyList.resize(v);
    }

    void addEdge(int v, int w)
    {
        adjacencyList[v].push_back(w);
        adjacencyList[w].push_back(v);
    }

    void parallelBFS(int startVertex)
    {
        vector<bool> visited(vertices, false);
        vector<int> frontier;
        vector<int> next_frontier;

        visited[startVertex] = true;
        frontier.push_back(startVertex);

        while (!frontier.empty())
        {
            next_frontier.clear();

#pragma omp parallel
            {
                vector<int> local_frontier;

#pragma omp for nowait
                for (size_t i = 0; i < frontier.size(); i++)
                {
                    int currentVertex = frontier[i];

                    for (int adjacentVertex : adjacencyList[currentVertex])
                    {
                        bool expected = false;
                        if (!visited[adjacentVertex])
                        {
#pragma omp critical
                            {
                                if (!visited[adjacentVertex])
                                {
                                    visited[adjacentVertex] = true;
                                    local_frontier.push_back(adjacentVertex);
                                }
                            }
                        }
                    }
                }

#pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(), local_frontier.begin(), local_frontier.end());
                }
            }
            frontier.swap(next_frontier);
        }
    }

    void parallelDFSHelper(int currentVertex, vector<bool> &visited)
    {
        for (int adjacentVertex : adjacencyList[currentVertex])
        {
            bool expected = false;
            if (!visited[adjacentVertex])
            {
#pragma omp critical
                {
                    if (!visited[adjacentVertex])
                    {
                        visited[adjacentVertex] = true;
                        expected = true;
                    }
                }
                if (expected)
                {
#pragma omp task
                    parallelDFSHelper(adjacentVertex, visited);
                }
            }
        }
    }

    void parallelDFS(int startVertex)
    {
        vector<bool> visited(vertices, false);
        visited[startVertex] = true;

#pragma omp parallel
        {
#pragma omp single
            {
                parallelDFSHelper(startVertex, visited);
            }
        }
    }
};

int main()
{
    int numVertices = 100000;
    int numEdges = 500000;
    Graph g(numVertices);

    for (int i = 0; i < numEdges; i++)
    {
        g.addEdge(rand() % numVertices, rand() % numVertices);
    }

    auto start = chrono::high_resolution_clock::now();
    g.parallelBFS(0);
    cout << "Parallel BFS Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    g.parallelDFS(0);
    cout << "Parallel DFS Time: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count() << " ms\n";

    return 0;
}