from collections import defaultdict

import pandas as pd
import numpy as np
import math
from scipy import stats
import json

if __name__ == '__main__':
    def reduce_matrix(graph: dict) -> dict:
        new_graph = graph.copy()
        visited = set()

        def dfs(n):
            visited.add(n)
            if n in graph:
                for neighbor in list(new_graph[n]):
                    if neighbor not in visited:
                        dfs(neighbor)
                    else:
                        # if cycle is detected
                        if neighbor in new_graph and new_graph[neighbor].get(n) is not None:
                            if new_graph[n][neighbor] < new_graph[neighbor][n]:
                                del new_graph[neighbor][n]
                            else:
                                del new_graph[n][neighbor]

        for node in new_graph:
            if node not in visited:
                dfs(node)

        return new_graph

    def get_max_edge(edges: dict) -> (str, str):
        max_key = None
        max_inner_key = None
        max_value = float('-inf')

        for key, inner_dict in edges.items():
            for inner_key, value in inner_dict.items():
                if value > max_value:
                    max_key = key
                    max_inner_key = inner_key
                    max_value = value

        return max_key, max_inner_key


    def create_dag(graph: dict) -> ([], dict):
        new_graph = graph.copy()
        edges = dict()
        visited = set()
        ordering = []

        def dfs(n):
            visited.add(n)

            # check whether node has outgoing edges
            if n in graph:
                for neighbor in list(new_graph[n]):
                    edges[n] = {}
                     # print(f"trying to insert {graph[n][neighbor]} into edges[{n}][{neighbor}]")
                    edges[n][neighbor] = graph[n][neighbor]
                    if neighbor not in visited:
                        dfs(neighbor)
                    else:
                        # if node has already been visited
                        # check whether neighbor has an outgoing edge
                        if neighbor in graph:
                            source, target = get_max_edge(edges)
                            # print(f"deleting new_graph[{source}][{target}]")
                            del new_graph[source][target]

            # if node has no outgoing edges
            else:
                print(f"edges: {edges} cleared")
                edges.clear()

            ordering.append(n)


        for node in new_graph:
            if node not in visited:
                dfs(node)

        return ordering[::-1], new_graph


    graph_dict = {
        'gender': {'education': 0.9586378320857879, 'marital': 0.9, 'income': 0.18379906315122246},
        'education': {'gender': 0.924441022694466, 'marital': 0.82, 'income': 0.1406747001721796},
        'marital': {'gender': 0.91, 'education': 0.8, 'hi': 1.0},
        'hi': {'gender': 1.1}
    }

    nested_dict = {"gender": {"marital": 0.9}, "marital": {"education": 0.8}}

    print(get_max_edge(nested_dict))
    reduced_matrix = reduce_matrix(graph_dict)
    print(reduced_matrix)
    ordering, dag = create_dag(reduced_matrix)
    print(f"ordering: {ordering}")
    print(dag)



