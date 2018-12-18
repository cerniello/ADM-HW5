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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from queue import Queue


class SnapGraph:
    def __init__(self, data_dir="./data/", verbose=True):
        self.data_dir = data_dir

        self.nodes = None
        self.edges = None
        self.graph = None
        self.graph_in = None

        self.node_degrees = None
        self.avg_node_degree = None

        self.categories = None
        self.page_names = None

        self.edges_fname = "wiki-topcats-reduced.txt"
        self.cat_fname = "wiki-topcats-categories.txt"
        self.page_names_fname = "wiki-topcats-page-names.txt"

        self.cpu_count = cpu_count()
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
            self._download_file(url, fname=self.page_names_fname + ".gz")
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
        # graph with outgoing edges for vertices
        graph_out = defaultdict(dict)
        # graph with incoming edges for vertices
        graph_in = defaultdict(dict)
        node_degrees = defaultdict(int)
        pbar = tqdm(edges.iterrows(), total=edges.shape[0])
        pbar.set_description("Building graph")
        for i, edge in pbar:
            v_s, v_e = edge
            # add a weighted edge (here weight = 1)
            graph_out[v_s][v_e] = 1
            graph_in[v_e][v_s] = 1
            node_degrees[v_s] += 1

        node_degrees = pd.DataFrame.from_dict(node_degrees, orient="index")
        self.node_degrees = node_degrees
        self.avg_node_degree = node_degrees.mean()
        self.nodes = nodes
        self.graph = graph_out
        self.graph_in = graph_in
        return

    def block_rank_category(self, inp_category):
        if self.categories is None:
            self.build_graph()

        if inp_category not in self.categories:
            raise ValueError(f"Provided category {inp_category} not found in database.")
        else:
            nodes_source_cat = self.categories[inp_category]

        block_rank_vec = heapdict()
        categ_sorted_nodes = dict()

        inf = float("inf")
        shortest_paths = defaultdict(list)
        n_cpus = self.cpu_count

        pbar = tqdm(total=len(nodes_source_cat))
        pbar.set_description("Computing for all source nodes")

        distances = []
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            dist_pools = list((executor.submit(self.dijkstra, src=n_src) for n_src in nodes_source_cat))
            for future in as_completed(dist_pools):
                pbar.update(1)
                distances.append(future.result())
        pbar.close()

        # loop through all dicts of each source node to gather the distances to
        # the categories
        for dists in distances:
            if dists is not None:  # if src node had out_edges
                for category, nodes in self.categories.items():
                    if category != inp_category:  # skip the input category for computation
                        for target in nodes:
                            if target in self.graph_in:  # only nodes with in_edges have dists
                                dist = dists[target]
                                if dist != inf:
                                    shortest_paths[category].append(dist)

        # compute the medians for all categories
        for category, nodes in self.categories.items():
            if category != inp_category:
                if shortest_paths:  # if there are articles that can be reached
                    block_rank_vec[category] = np.median(shortest_paths[category])
                else:
                    block_rank_vec[category] = inf

        # initiate the sorting algorithm. The loaded edges dataframe will be updated inplace
        # with every call to _create_score_sort. No direct subgraph computation happening.
        # The 'already included' nodes will be marked with 1 on 'in_sub'.
        edges = self.load_data_edges(self.edges_fname)
        edges["score"] = 1
        edges["in_sub"] = 0
        categ_sorted_nodes[inp_category] = self._create_score_sort(inp_category, edges)
        for categ in self.categories.keys():
            if categ != inp_category:
                categ_sorted_nodes[categ] = self._create_score_sort(categ, edges)

        output_list = []
        while block_rank_vec:
            category, dist = block_rank_vec.popitem()
            # output elements will be tuples of
            # (Category, Rank, Sorted_Nodes_list)
            output_list.append((category, dist, categ_sorted_nodes[category]))
        return output_list

    def _create_score_sort(self, category, edges_score):
        # get all the nodes in the category 0
        nodes_in_cat = self.categories[category]
        rel_edges = edges_score[((edges_score["v_end"].isin(nodes_in_cat)) &
                                 (edges_score["v_start"].isin(nodes_in_cat))) |
                                ((edges_score["in_sub"] == 1) &
                                 (edges_score["v_end"].isin(nodes_in_cat)))]

        scores = rel_edges.loc[:, ["v_end", "score"]].groupby(by=["v_end"]).sum()["score"]

        scores.name = "score"

        edges_score.set_index("v_end", inplace=True)
        edges_score.loc[scores.index, "score"] = scores
        edges_score.loc[scores.index, "in_sub"] = 1
        edges_score.reset_index(inplace=True)

        return scores.sort_values(ascending=False)

    def _compute_distances(self, inp_category):
        # init node dicts
        visited = {}  # visited = True/False
        parent = {}  # parent of the node
        distance = {}  # distance from C0

        inf = float('inf')
        # init the dict for each node in self.nodes
        for v in self.graph_in.keys():
            visited[v] = False
            parent[v] = None
            distance[v] = inf

        pbar = tqdm(self.categories[inp_category])
        # for each node in self.categories[inp_category]
        for v in pbar:
            # create a queue
            Q = Queue()

            # put the node in the queue
            # each C0 node has 0 as distance!
            Q.put(v)
            visited[v] = True
            dist = 0
            distance[v] = dist

            # if the queue is not empty
            while not Q.empty():
                # pick the next node
                u = Q.get()
                # increase the distance
                dist += 1
                # for each out_edge of u
                for z in self.graph[u].keys():

                    # if it's not visited update the values
                    if not visited[z]:
                        Q.put(z)
                        visited[z] = True
                        parent[z] = u
                        distance[z] = dist
                    else:
                        # if the distance from the new c0 node
                        # is less than the previous one
                        # update the distance and the parent
                        if distance[z] > dist:
                            Q.put(z)
                            parent[z] = u
                            distance[z] = dist

        return distance

    def dijkstra(self, src):
        """
        Compute the shortest distance from the source vertex to alll other nodes
        according to the dijkstra algorithm.

        :param src: int, the number of the source vertex
        :return: dict, the distances of each vertex to source
        """

        if src not in self.graph.keys():
            return None

        inf = float("inf")
        # a heap dictionary to keep track of the next closest
        # vertex in the nodes. Serves as a priority queue to speed
        # up the algorithm.
        dist_heap = heapdict()
        for v in self.graph_in.keys():
            dist_heap[v] = inf
        dist_heap[src] = 0

        final_dists = dict()

        while True:

            # Pick minimum distance vertex from
            # unprocessed vertices
            # u = src in first iteration
            u, dist = dist_heap.popitem()

            # the dist to u is now final
            final_dists[u] = dist

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

            # if priority queue is empty
            if not dist_heap:
                break

        return final_dists

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
        with open(self.data_dir + fname, write_mode) as f:
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
    snap_graph.block_rank_category('American_Jews')
