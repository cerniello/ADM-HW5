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


# nodes = list(range(100))
# graph = dict()
# for k in nodes:
#     graph[k] = dict()
#
# for i in nodes:
#     for j in nodes:
#         if np.random.choice([0,1], p=[0.5, 0.5]):
#             graph[i][j] = 1
#
#
# def dijkstra(src, trgts=None):
#     """
#     Compute the shortest distance from the source vertex to the target
#     according to the dijkstra algorithm.
#
#     :param src: int, the number of the source vertex
#     :param trgts: list, the number of the target vertices that are wanted.
#                         If None, all distances will be calculated.
#     :return: int, the distance
#     """
#
#     if src not in nodes:
#         return [float("inf")] * len(trgts)
#
#     trgts_dists = dict()
#     rem_trgts = []
#     # some articles do not have any edge going in or going out.
#     # -> Unreachable
#     for trgt in trgts:
#         if trgt not in nodes:
#             trgts_dists[trgt] = float("inf")
#         else:
#             rem_trgts.append(trgt)
#
#     # a heap dictionary to keep track of the next closest
#     # vertex in the nodes. Serves as a priority queue to speed
#     # up the algorithm.
#     dist_heap = heapdict()
#     for v in nodes:
#         dist_heap[v] = float("inf")
#     dist_heap[src] = 0
#
#     final_dists = dict()
#
#     for _ in range(len(nodes)):
#
#         # Pick minimum distance vertex from
#         # unprocessed vertices
#         # u = src in first iteration
#         u, dist = dist_heap.popitem()
#
#         # the dist to u is now final
#         final_dists[u] = dist
#         if u in rem_trgts:
#             trgts_dists[u] = dist
#             rem_trgts.remove(u)
#
#         # if all target distances have been found
#         if not rem_trgts:
#             break
#
#         # Update dist value of the adjacent vertices
#         # of the picked vertex only if the current
#         # distance is greater than new distance and
#         # the vertex is not in the shortest path tree
#         for v in graph[u]:
#             if v not in final_dists:
#                 dist_add = graph[u][v]
#                 if dist_add > 0:
#                     new_dist = dist + dist_add
#                     if dist_heap[v] > new_dist:
#                         dist_heap[v] = new_dist
#
#     return [trgts_dists[trgt] for trgt in trgts], _
#
#
# print(dijkstra(np.random.choice(nodes), np.random.choice(nodes, size=4)))


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

    @staticmethod
    def get_num_lines(file):
        lines = 0
        while file.readline():
            lines += 1
        return lines

    def graph_built(self):
        return all([x is not None for x in (self.nodes,
                                            self.edges,
                                            self.graph,
                                            self.node_degrees,
                                            self.avg_node_degree)])

    def load_data_edges(self, fname):
        edges = self.data_dir + fname
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
        return edges

    def load_data_all(self, edges_fname=None, categs_fname=None, pagenames_fname=None):
        if edges_fname is None:
            edges = self.edges_fname
        else:
            edges = self.data_dir + edges_fname
        if categs_fname is None:
            cat = self.data_dir + self.cat_fname
        else:
            cat = self.data_dir + categs_fname
        if pagenames_fname is None:
            page_names = self.data_dir + self.page_names_fname
        else:
            page_names = self.data_dir + pagenames_fname

        edges = self.load_data_edges(edges)
        self.edges = edges

        if isfile(cat):
            f = open(cat)
        elif isfile(cat + ".gz"):
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(cat + ".gz")))

        else:
            url = "https://snap.stanford.edu/data/wiki-topcats-categories.txt.gz"
            self._download_file(url, fname=self.cat_fname + ".gz")
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(cat + ".gz")))

        total = SnapGraph.get_num_lines(f)
        f.seek(0)
        cats = self._read_category_file(f, total=total)
        self.categories = cats

        if isfile(page_names):
            f = open(page_names)
        elif isfile(page_names + ".gz"):
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(page_names + ".gz")))
        else:
            url = "https://snap.stanford.edu/data/wiki-topcats-page-names.txt.gz"
            self._download_file(url, fname=self.page_names_fname)
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(page_names + ".gz")))

        total = SnapGraph.get_num_lines(f)
        f.seek(0)
        page_names = pd.DataFrame(self._read_pagenames_file(f, total=total), columns=["name"])
        self.page_names = page_names

        return edges, cats, page_names

    @staticmethod
    def _read_category_file(file, **kwargs):
        cats = dict()
        pbar = tqdm(file, **kwargs)
        pbar.set_description("Processing categories")
        for line in pbar:
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
    def _read_pagenames_file(file, **kwargs):
        page_names = []
        pbar = tqdm(file, **kwargs)
        pbar.set_description("Processing page names")
        for line in pbar:
            name = line.split(" ", 1)[1].strip()
            page_names.append(name)

        file.close()

        return page_names

    def build_graph(self):
        edges, cats, name = self.load_data_all()
        nodes = pd.unique(pd.concat((edges["v_start"], edges["v_end"])))
        graph = defaultdict(dict)
        node_degrees = defaultdict(int)
        pbar = tqdm(edges.iterrows(), total=edges.shape[0])
        pbar.set_description("Building graph")
        for i, edge in pbar:
            v_s, v_e = edge
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

        block_rank_vec = heapdict()
        categ_sorted_nodes = dict()

        pbar = tqdm(self.categories)
        for cat in pbar:
            pbar.set_description("Processing category %s" % cat)
            if cat == inp_category:
                continue
            nodes_targ_cat = self.categories[cat]
            shortest_paths = []
            for n_st in nodes_source_cat:
                dists = self.dijkstra(src=n_st, trgts=nodes_targ_cat)
                for dist in dists:
                    if dist != float("inf"):
                        # if the node is unreachable he will be excluded from the
                        # dist calc
                        shortest_paths.append(dist)
            if shortest_paths:  # if there are articles that can be reached
                block_rank_vec[cat] = np.median(shortest_paths)
            else:
                block_rank_vec[cat] = float("inf")

        edges = self.load_data_edges(self.edges_fname)
        edges["score"] = 1
        for categ in [inp_category] + list(self.categories.keys()):
            categ_sorted_nodes[categ] = self._create_score_sort(categ, edges)

        output_list = []
        while block_rank_vec:
            category, dist = block_rank_vec.popitem()
            output_list.append((category, dist, categ_sorted_nodes[category]))
        return output_list

    def _create_score_sort(self, category, edges_score):
        # get all the nodes in the category 0
        nodes_in_cat = self.categories[category]
        rel_edges = edges_score[edges_score["v_end"].isin(nodes_in_cat)]

        scores = rel_edges.groupby(by=["v_end"]).sum()["score"]

        scores.name = "score"

        edges_score.set_index("v_end", inplace=True)
        edges_score.loc[scores.index, "score"] = scores
        edges_score.reset_index(inplace=True)

        return scores.sort_values(ascending=False)

    def dijkstra(self, src, trgts=None):
        """
        Compute the shortest distance from the source vertex to the target
        according to the dijkstra algorithm.

        :param src: int, the number of the source vertex
        :param trgts: list, the number of the target vertices that are wanted.
                            If None, all distances will be calculated.
        :return: int, the distance
        """

        if src not in self.nodes:
            return [float("inf")] * len(trgts)

        trgts_dists = dict()
        rem_trgts = []
        # some articles do not have any edge going in or going out.
        # -> Unreachable
        for trgt in trgts:
            if trgt not in self.nodes:
                trgts_dists[trgt] = float("inf")
            else:
                rem_trgts.append(trgt)

        # a heap dictionary to keep track of the next closest
        # vertex in the nodes. Serves as a priority queue to speed
        # up the algorithm.
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
            if u in rem_trgts:
                trgts_dists[u] = dist
                rem_trgts.remove(u)

            # if all target distances have been found
            if not rem_trgts:
                break

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex is not in the shortest path tree
            for v in self.graph[u]:
                if v not in final_dists:
                    dist_add = self.graph[u][v]
                    if dist_add > 0:
                        new_dist = dist + dist_add
                        if dist_heap[v] > new_dist:
                            dist_heap[v] = new_dist

        return [trgts_dists[trgt] for trgt in trgts]

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
    snap_graph.block_rank_category("English_footballers")
