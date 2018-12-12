import numpy as np
import pandas as pd
import os
from os.path import isfile
import requests
from tqdm import tqdm
import gzip
import io


class SnapGraph:
    def __init__(self, data_dir="./data/", verbose=True):
        self.data_dir = data_dir
        self.nodes = None
        self.edges = None

        self.avg_node_degree = None

        self.graph_data_fname = "wiki-topcats-reduced.txt"
        self.cat_fname = "wiki-topcats-categories.txt"
        self.page_names_fname = "wiki-topcats-page-names.txt"

        self.verbose = verbose

    def load_data(self):
        graph_data = self.data_dir + self.graph_data_fname
        cat = self.data_dir + self.cat_fname
        page_names = self.data_dir + self.page_names_fname

        if isfile(graph_data):
            compression = None
        elif isfile(graph_data + ".zip"):
            graph_data = graph_data + ".zip"
            compression = "zip"
        else:
            url = "https://doc-0k-4g-docs.googleusercontent.com/docs" \
                  "/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/54sksu9ea" \
                  "4b4213pa19ivh7v05p1ekhf/1544529600000/00993335131002784161/" \
                  "*/1ghPJ4g6XMCUDFQ2JPqAVveLyytG8gBfL?e=download"
            self._download_file(url, fname=self.graph_data_fname)
            graph_data = graph_data + ".zip"
            compression = "zip"

        graph_data = pd.read_csv(graph_data, sep="\t",
                                 header=None, index_col=None,
                                 compression=compression)
        graph_data.columns = ["category", "index"]

        if isfile(cat):
            f = open(cat)
        elif isfile(cat + ".gz"):
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(cat + ".gz")))
        else:
            url = "https://snap.stanford.edu/data/wiki-topcats-categories.txt.gz"
            self._download_file(url, fname=self.cat_fname + ".gz")
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(cat + ".gz")))

        cats = self._read_category_file(f)

        if isfile(page_names):
            f = open(page_names)
        elif isfile(page_names + ".gz"):
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(page_names + ".gz")))
        else:
            url = "https://snap.stanford.edu/data/wiki-topcats-page-names.txt.gz"
            self._download_file(url, fname=self.page_names_fname)
            f = io.TextIOWrapper(io.BufferedReader(gzip.open(page_names + ".gz")))

        page_names = pd.DataFrame(self._read_pagenames_file(f), columns=["name"])

        return graph_data, cats, page_names

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
        g, c, p = self.load_data()

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
    graph = SnapGraph()
    graph.build_graph()