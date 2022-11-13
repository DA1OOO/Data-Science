import pandas as pd
import networkx as nx

DIMENSION = 54
MY_ID = 8
df = pd.read_csv('Social_Matrix_2022_9.csv', header=None)
sociomatrix = df.values

directed_graph = nx.DiGraph()
directed_graph.add_nodes_from(range(1, DIMENSION + 1))
for i in range(0, DIMENSION):
    for j in range(0, len(sociomatrix[0])):
        if sociomatrix[i][j] == 1:
            directed_graph.add_edge(i + 1, j + 1)

hubs, authorities = nx.hits(directed_graph)
print("My in degree is {}".format(directed_graph.in_degree(MY_ID)))
print("My out degree is {}".format(directed_graph.out_degree(MY_ID)))
print("My closeness centrality is {}".format(nx.closeness_centrality(directed_graph, MY_ID)))
print("My betweenness centrality is {}".format(nx.betweenness_centrality(directed_graph)[MY_ID]))
print("My hubness ranking is {}".format(hubs[MY_ID]))
print("My authority ranking is {}".format(authorities[MY_ID]))
