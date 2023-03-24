"""Loads and represents graph datasets from the TU repository.

A TUDataset is a collection of graphs such that each graph is the "record" of
a natural person. Such a graph could be, e.g., the k-IIG of a person in a
social network, or internal cash flows of a company.

For datasets where user information lies in a single node or edge, see
tapas.datasets.network instead.

"""

import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import requests
import zipfile

from .dataset import RecordSetDataset, Record, DataDescription
from .utils import index_split


## Helper functions for TU Datasets.


def _download_url(url, fp):
    """
    Download the content (a TU dataset) from a given url.

    """
    if os.path.isdir(fp):
        path = os.path.join(fp, url.split("/")[-1])
    else:
        path = fp

    print("Downloading %s from %s..." % (path, url))

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError("Failed downloading url %s" % url)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    return path


def _process(name, filepath):
    """
    Parse files in TU format into a TUDataset object.

    """
    print("Loading TU graph dataset: " + str(name))
    graph = nx.Graph()

    # add edges
    data_adj = np.loadtxt(
        os.path.join(filepath, "{}_A.txt".format(name)), delimiter=","
    ).astype(int)
    data_tuple = list(map(tuple, data_adj))
    graph.add_edges_from(data_tuple)

    # If provided, add edge labels.
    f = os.path.join(filepath, "{}_edge_labels.txt".format(name))
    if os.path.exists(f):
        data_edge_labels = np.loadtxt(f, delimiter=",").astype(int).tolist()
        nx.set_edge_attributes(graph, dict(zip(data_tuple, data_edge_labels)), "label")

    # If provided, add edge attributes.
    f = os.path.join(filepath, "{}_edge_attributes.txt".format(name))
    if os.path.exists(f):
        data_edge_attributes = np.loadtxt(f, delimiter=",").tolist()
        nx.set_edge_attributes(
            graph, dict(zip(data_tuple, data_edge_attributes)), "attribute"
        )

    # If provided, add nodes, their labels and attributes.
    f = os.path.join(filepath, "{}_node_labels.txt".format(name))
    if os.path.exists(f):
        data_node_label = np.loadtxt(f, delimiter=",").astype(int).tolist()
    else:
        # Otherwise, the label of a node is its internal designation.
        # Note that this is somewhat pointless, but required for some operations.
        data_node_label = list(graph.nodes)

    f = os.path.join(filepath, "{}_node_attributes.txt".format(name))
    has_node_attr = False
    if os.path.exists(f):
        data_node_attribute = np.loadtxt(f, delimiter=",").tolist()
        has_node_attr = True

    for i in range(len(data_node_label)):
        graph.add_node(
            i + 1,
            label=data_node_label[i],
            attribute=data_node_attribute[i] if has_node_attr else None,
        )

    # The following files are required in the TU format.
    data_graph_indicator = np.loadtxt(
        os.path.join(filepath, "{}_graph_indicator.txt".format(name)), delimiter=","
    ).astype(int)

    data_graph_label = np.loadtxt(
        os.path.join(filepath, "{}_graph_labels.txt".format(name)), delimiter=","
    ).astype(int)

    has_graph_attr = False
    f = os.path.join(filepath, "{}_graph_attributes.txt".format(name))
    if os.path.exists(f):
        data_graph_attribute = np.loadtxt(f, delimiter=",")
        has_graph_attr = True

    # split into sub-graphs using graph indicator
    graphs = []
    for idx in range(data_graph_indicator.max()):
        node_idx = np.where(data_graph_indicator == (idx + 1))
        node_list = [x + 1 for x in node_idx[0]]

        sub_graph = graph.subgraph(node_list)
        sub_graph.graph["label"] = data_graph_label[idx]
        sub_graph.graph["attribute"] = (
            data_graph_attribute[idx] if has_graph_attr else None
        )
        graphs.append(sub_graph)

    # pandas.Series of networkx objects
    graphs = pd.Series(graphs)
    description = TUDatasetDescription(label=name)

    return TUDataset(graphs, description)


## Instances of the different classes for TU datasets.


# The description is identical.
class TUDatasetDescription(DataDescription):
    """
    TU Datasets all have the same format. This description class thus only
    contains a label. This is required for compatibility with other classes,
    i.e. threat models and some of the functionalities of TabularDatasets.

    """


class TUDataset(RecordSetDataset):
    """
    Class to represent TU network data as a Dataset. Internally, the data
    is stored as Pandas Series of networkx objects.

    """

    _url = r"https://www.chrsmrrs.com/graphkerneldatasets/"

    def __init__(self, data, description):
        """
        Parameters
        ----------
        data: pandas.Series
        description: tapas.datasets.TUDatasetDescription

        """
        assert isinstance(
            description, TUDatasetDescription
        ), "description needs to be of class DataDescription"
        # Ininitalise the parent class with the correct record class.
        RecordSetDataset.__init__(self, data, description, TURecord)

    @classmethod
    def read(cls, name, root=None):
        """
        Process TU files

        Parameters
        ----------
        name: str
            The name of the dataset.
        root: str (default: use name)
            root directory where the dataset is saved

        Returns
        -------
        TUDataset
            A TUDataset.

        """
        root = root if root is not None else name
        return _process(name, root)

    @classmethod
    def download_and_read(cls, name, root=None):

        """
        Download and process TU files

        Parameters
        ----------
        name: str
            The name of the dataset.
        root: str
            root directory where the dataset to be saved

        Returns
        -------
        TUDataset
            A TUDataset.

        """
        if root is None:
            root = "./"
        filepath = _download_url(f"{cls._url}/{name}.zip", root)
        with zipfile.ZipFile(filepath, "r") as f:
            f.extractall(root)

        filepath = os.path.join(root, name)
        return _process(name, filepath)

    def write(self, output_path):
        """
        Write dataset as object to file.

        """
        with open(output_path, "wb") as f:
            pickle.dump(self.data, f)

    def read_from_string(self, data, schema):
        raise NotimplementedError()

    def write_to_string(self):
        raise NotimplementedError()


class TURecord(Record, TUDataset):
    """
    Class for TU record object. The TU data is a 1D array

    """

    def __init__(self, data, description, identifier):
        Record.__init__(self, data, description, identifier)
        TUDataset.__init__(self, data, description)

    @classmethod
    def from_dataset(cls, tu_dataset):
        """
        Create a TUDataset object from a TUDataset object containing 1 record.

        Parameters
        ----------
        tu_dataset: TUDataset
            A TUDataset object containing one record.

        Returns
        -------
        TUDataset
            A TUDataset object

        """
        if tu_dataset.data.shape[0] != 1:
            raise AssertionError(
                f"Parent TUDataset object must contain only 1 record, not {tu_dataset.data.shape[0]}"
            )

        return cls(
            tu_dataset.data, tu_dataset.description, tu_dataset.data.index.values[0]
        )
