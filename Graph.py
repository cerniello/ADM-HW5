import numpy as np
import pandas as pd
import os
from os.path import isfile
import requests
from tqdm import tqdm
import gzip
import io
from collections import defaultdict
from heapdict import heapdict


class SnapGraph:
    def __init__(self, data_dir="./data/", verbose=True):
        self.data_dir = data_dir

        self.nodes = None
        self.edges = None
        self.graph = None

        self.node_degrees = None
        self.avg_node_degree = None

        self.categories = None
        self.page_names = None

        self.edges_fname = "wiki-topcats-reduced.txt"
        self.cat_fname = "wiki-topcats-categories.txt"
        self.page_names_fname = "wiki-topcats-page-names.txt"

        self.verbose = verbose

    def graph_built(self):
        return all([x is not None for x  in (self.nodes,
                                             self.edges,
                                             self.graph,
                                             self.node_degrees,
                                             self.avg_node_degree)])

    def load_data(self):
        edges = self.data_dir + self.edges_fname
        cat = self.data_dir + self.cat_fname
        page_names = self.data_dir + self.page_names_fname

        if isfile(edges):
            compression = None
        elif isfile(edges + ".zip"):
            edges = edges + ".zip"
            compression = "zip"
        else:
            url = "https://doc-0k-4g-docs.googleusercontent.com/docs" \
                  "/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/54sksu9ea" \
                  "4b4213pa19ivh7v05p1ekhf/1544529600000/00993335131002784161/" \
                  "*/1ghPJ4g6XMCUDFQ2JPqAVveLyytG8gBfL?e=download"
            self._download_file(url, fname=self.edges_fname)
            edges = edges + ".zip"
            compression = "zip"

        edges = pd.read_csv(edges, sep="\t",
                            header=None, index_col=None,
                            compression=compression)
        edges.columns = ["v_start", "v_end"]
        self.edges = edges

        if isfile(cat):
            f = open(cat)
        elif isfile(cat + ".gz"):
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(cat + ".gz")))
        else:
            url = "https://snap.stanford.edu/data/wiki-topcats-categories.txt.gz"
            self._download_file(url, fname=self.cat_fname + ".gz")
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(cat + ".gz")))

        cats = self._read_category_file(f)
        self.categories = cats

        if isfile(page_names):
            f = open(page_names)
        elif isfile(page_names + ".gz"):
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(page_names + ".gz")))
        else:
            url = "https://snap.stanford.edu/data/wiki-topcats-page-names.txt.gz"
            self._download_file(url, fname=self.page_names_fname)
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(page_names + ".gz")))

        page_names = pd.DataFrame(self._read_pagenames_file(f), columns=["name"])
        self.page_names = page_names

        return edges, cats, page_names

    @staticmethod
    def _read_category_file(file):
        cats = dict()
        for line in file:
            rem_cat_word = line.split(":", 1)[1]
            category, indices = rem_cat_word.split(";", 1)
            indices = indices.strip()
            if indices:
                indices = list(map(int, indices.split(" ")))
            else:
                indices = []
            if len(indices) > 3500:
                cats[category] = indices

        file.close()

        return cats

    @staticmethod
    def _read_pagenames_file(file):
        page_names = []
        for line in file:
            name = line.split(" ", 1)[1].strip()
            page_names.append(name)

        file.close()

        return page_names

    def build_graph(self):
        edges, cats, name = self.load_data()
        nodes = set()
        graph = defaultdict(dict)
        node_degrees = defaultdict(int)
        for i, edge in tqdm(edges.iterrows(), total=edges.shape[0]):
            v_s, v_e = edge
            nodes.update((v_s, v_e))
            # add a weighted edge (here weight = 1)
            graph[v_s][v_e] = 1
            node_degrees[v_s] += 1

        node_degrees = pd.DataFrame.from_dict(node_degrees, orient="index")
        self.node_degrees = node_degrees
        self.avg_node_degree = node_degrees.mean()
        self.nodes = nodes
        self.graph = graph
        return

    def block_rank_category(self, inp_category):
        if inp_category not in self.categories:
            raise ValueError(f"Provided category {inp_category} not found in database.")
        else:
            nodes_source_cat = self.categories[inp_category]
        if self.categories is None:
            self.build_graph()

        c_0_subgraph = self._create_subgraph(inp_category)

        block_rank_vec = heapdict()

        for cat in self.categories:
            nodes_targ_cat = self.categories[cat]
            shortest_paths = []
            for n_st in nodes_source_cat:
                for n_end in nodes_targ_cat:
                    shortest_paths.append(self.dijkstra(src=n_st, trgt=n_end))

            block_rank_vec[cat] = - np.median(shortest_paths)

    def _create_subgraph(self, c_0):
        # get all the nodes in the category 0
        nodes_in_cat = self.categories[c_0]
        sub_graph = dict()
        scores = defaultdict(int)
        for v in self.nodes:
            sub_graph[v] = dict()
            cat_neighb = set(self.graph[v]).intersection(nodes_in_cat)
            for n in cat_neighb:
                w = self.graph[v][n]
                scores[n] += w
                sub_graph[v][n] = w

        return sub_graph, scores

    def _extend_subgraph(self, c_i, sub_graph, scores):
        """
        Extend a subgraph by another category and compute scores

        :param sub_graph: dict, the current subgraph
        :param scores: defaultdict, the scores of the nodes so far
        :param c_i: str, the name of the next category to extend with
        :return: dict, the extended subgraph
        """

        nodes_in_cat = self.categories[c_i]
        for v in self.nodes:
            cat_neighb = set(self.graph[v]).intersection(nodes_in_cat)
            for n in cat_neighb:
                if v in sub_graph:
                    # increase incoming score of neighbour by score of sender
                    w = scores[v]
                else:
                    w = self.graph[v][n]

                scores[n] += w
                sub_graph[v][n] = w

    def dijkstra(self, src, trgt):
        """
        Compute the shortest distance from the source vertex to the target
        according to the dijkstra algorithm.

        :param src: int, the number of the source vertex
        :param trgt: int, the number of the target vertex
        :return: int, the distance
        """

        # a heap dictionary to keep track of the next closest
        # vertex in the nodes
        dist_heap = heapdict()
        for v in self.nodes:
            dist_heap[v] = float("inf")
        dist_heap[src] = 0

        final_dists = dict()

        for _ in range(len(self.nodes)):

            # Pick minimum distance vertex from
            # unprocessed vertices
            # u = src in first iteration
            u, dist = dist_heap.popitem()

            # the dist to u is now final
            final_dists[u] = dist
            if u == trgt:
                break

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex is not in the shortest path tree
            for v in self.graph[u]:
                if v in dist_heap:
                    dist_add = self.graph[u][v]
                    if dist_add > 0:
                        new_dist = dist_heap[u] + dist_add
                        if dist_heap[v] > new_dist:
                            dist_heap[v] = new_dist

        return final_dists[trgt]

    def _download_file(self, url, fname=None, write_mode='wb', **kwargs):
        """
        Download routine for a given url. Can resume download if previously attempted
        and not fully completed yet via simple filesize check (might be error prone).

        :param url: string, url link
        :param write_mode: string, "wb" for write in byte mode, "ab" for append in byte mode
        :param kwargs: kwargs passed to the requests.get() function
        :return: None
        """

        if fname is None:
            fname = url.split("/")[-1]

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True, **kwargs)

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        with open(self.data_dir + fname + ".txt", write_mode) as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for data in r.iter_content(32 * 1024):
                    f.write(data)
                    pbar.update(len(data))
        self._print("Download finished.")
        return

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
        return


if __name__ == '__main__':
    snap_graph = SnapGraph()
    snap_graph.build_graph()
