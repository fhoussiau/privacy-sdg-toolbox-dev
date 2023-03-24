"""Classes to represent a dataset as a single network.

In this file, we consider datasets where the information of a single use
is represented by a node or edge on a graph. We use the networkx library
to represent such graph datasets.

For datasets where the user data is a graph, see tapas.datasets.tu_dataset.

"""

from .dataset import Dataset

import networkx as nx
import numpy as np


class NetworkDataset(Dataset):
    """
    A dataset where each user is represented either by a node or edge.

    The data is represented internally as a networkx.Graph object.
    This object can treat user data as either on a node or an edge, as
    specified by the mode attribute ("node" or "edge").

    When user records are extracted from this dataset, these take the form of 
    NodeRecord or EdgeRecord objects. This is because nodes/edges do not make
    sense in isolation, and these objects provide the context necessary to
    interpret and manipulate them.

    The network can also include node and edge attributets
    In principle, the network can include node and edge attributes.
    can include node and edge attributes (networkx allows it)

    """

    NODE = "node"
    EDGE = "edge"

    def __init__(self, data, mode="node", label=None):
        # Check that the input is correctly formatted.
        assert isinstance(data, nx.Graph), "Data is not a networkx Graph."
        assert mode in (NetworkDataset.NODE, NetworkDataset.EDGE), "Unknown mode."
        # Set internal variables.
        self.data = data
        self.mode = mode
        self._label = label or "Unnamed Dataset"

    @classmethod
    def read(cls, input_path, mode="node", label=None):
        """
        Read the network data from the specified path, returns a NetworkDataset.

        By default, this uses the adjacency list format (see the network doc at
        https://networkx.org/documentation/stable/reference/readwrite/adjlist.html)

        """
        G = nx.read_adjlist(input_path)
        label = label or input_path
        return cls(G, mode, label)

    def write(self, output_path):
        """
        Write the network data to the specified path.

        By default, this uses the adjacency list format (see the network doc at
        https://networkx.org/documentation/stable/reference/readwrite/adjlist.html)

        """
        nx.write_adjlist(self.data, output_path)

    def sample(self, n_users=1, random_state="ignored"):
        """
        Returns a list of n_users NodeRecords (if mode == "node", EdgeRecords 
        otherwise) sampled uniformly at random from the dataset.

        """
        if self.mode == NetworkDataset.NODE:
            records = self.data.nodes
            record_class = NodeRecord
        else:
            records = self.data.edges
            record_class = EdgeRecord
        # Select n_users random records from this list of records.
        records = list(records)
        indices = np.random.choice(len(records), replace=False, size=n_users)
        return [record_class(self, records[idx]) for idx in indices]

    def copy(self):
        """
        Create a Dataset that is a deep copy of this one. In particular,
        the underlying data is copied and can thus be modified freely.

        Returns
        -------
        Dataset
            A copy of this Dataset.

        """
        return NetworkDataset(self.data.copy(), self.mode, self.label)

    def add_records(self, records, in_place=True):
        """
        Add one or more nodes/edges to this graph.

        Contrarily to tapas.datasets.TabularDataset, this *does not* take a
        Dataset as attribute, but an iterable (or a single object) of NodeRecord
        and/or EdgeRecord.

        Parameters
        ----------
        records : NodeRecord, EdgeRecord, or iterable of such objects.
            The records to add to this dataset.
        in_place : bool (default True)
            Bool indicating whether or not to change the dataset in-place or return
            a copy. If True, the dataset is changed in-place. The default is False.

        Returns
        -------
        Dataset or None
            A new Dataset object with the record(s) or None if inplace=True.


        """
        # If only one record is provided, wrap it in a list.
        if isinstance(records, NodeRecord) or isinstance(records, EdgeRecord):
            records = [records]
        # If not in place, create a copy of this dataset.
        dataset = self if in_place else self.copy()
        graph = dataset.data
        # Then, add each record iteratively. Note that it is allowed to have
        # a mix of NodeRecord and EdgeRecord in the list, but this can create
        # unwanted behaviour (e.g. adding an edge before one of its adjacent
        # nodes). This should not happen in practice.
        # Also, this requires undirected graphs, so that adding nodes iteratively
        # does not create issues (since adding a node also adds its adjacent edges).
        for record in records:
            if isinstance(record, NodeRecord):
                # If the node is already in the data, this does nothing (but works).
                graph.add_node(record.node)
                # If any of the nodes do not exist, they are added to the graph.
                graph.add_edges_from(record.edges)
            elif isinstance(record, EdgeRecord):
                graph.add_edges_from([record.edge])
            else:
                raise Exception(f"Record type not accepted: {type(record)}.")
        # Finally, return the new dataset object if not in place.
        if not in_place:
            return dataset



    def __add__(self, other):
        """
        Compose two graphs together as one larger graph. This preserves all
        edges and nodes. If there are nodes in the same label in both graphs,
        these nodes are assumed to be the same.

        """
        assert isinstance(
            other, NetworkDataset
        ), "Can only add NetworkDatasets together."
        composed_data = self.data.compose(self.data, other.data)
        return NetworkDataset(composed_data, self.mode, self.label)

    def __iter__(self):
        """
        Returns an iterator over user records in the dataset.

        If mode == "node", then this returns an iterator over NodeRecords for
        each node in the dataset (and vice versa for "edge" and EdgeRecord).

        """
        if self.mode == NetworkDataset.NODE:
            iterator = self.data.nodes
            record_class = NodeRecord
        else:
            iterator = self.data.edges_iter()
            record_class = EdgeRecord
        # Create a record for each user, and yield.
        for user in iterator:
            yield record_class(self, user)

    def __len__(self):
        """
        Returns the number of users in the dataset.

        If mode == "node", this is the number of nodes, otherwise the number of edges.

        """
        if self.mode == NetworkDataset.NODE:
            return len(self.data.nodes)
        else:
            return self.data.number_of_edges()

    @property
    def label(self):
        return self._label


class NodeRecord:
    """A node extracted from a graph and its adjacent edges."""

    def __init__(self, dataset, node):
        self.node = node
        self.edges = list(dataset.data.edges(self.node))
        self._label = f"{dataset.label}->{node}"

    @property
    def label(self):
        return self._label

    # These are required for compatibility with RecordSetDatasets, where
    # records are treated as (iterable) datasets of size 1.
    def __iter__(self):
        yield self
    def __len__(self):
        return 1
    

class EdgeRecord:
    """An edge extracted from a graph."""

    def __init__(self, dataset, edge):
        self.edge = edge
        self._label = f"{dataset.label}->{edge}"

    @property
    def label(self):
        return self._label

    # These are required for compatibility with RecordSetDatasets, where
    # records are treated as (iterable) datasets of size 1.
    def __iter__(self):
        yield self
    def __len__(self):
        return 1