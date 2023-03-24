from tapas.datasets import NetworkDataset
from tapas.threat_models import TargetedMIA, ExactDataKnowledge, BlackBoxKnowledge

from tapas.attacks import SetFeature

import networkx as nx
import numpy as np

# Dummy handwritten dataset.
G = nx.Graph()
G.add_edges_from(
    [
        (0, 1),
        (1, 2),
        (3, 4),
        (2, 4),
        (2, 3),
        (0, 5),
        (0, 3),
        (2, 10),
        (9, 10),
        (3, 9),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 4),
        (8, 3),
    ]
)

# Create a dataset object.
dataset = NetworkDataset(G, mode="edge", label="My Dataset")

# Select a random target in this dataset, and remove it from G.
target = dataset.sample(1)[0]

# G.remove_node(target.node)
G.remove_edge(*target.edge)


# Slightly modified Raw generator (does not restrict output size).
class Raw:
    def __init__(self):
        super().__init__()

    def fit(self, dataset):
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples=None, random_state=None):
        return self.dataset

    def __call__(self, dataset, num_samples, random_state=None):
        self.fit(dataset)
        return self.generate(num_samples, random_state=random_state)

    @property
    def label(self):
        return "Raw"


# Instantiate the threat model, as usual. For this object, it works :tm:.
threat_model = TargetedMIA(
    ExactDataKnowledge(dataset),
    target,
    BlackBoxKnowledge(Raw(), len(dataset)),
    generate_pairs=True,
    replace_target=False,
)

# Instantiate an attack ... now what
from tapas.attacks import ShadowModellingAttack, FeatureBasedSetClassifier
from sklearn.ensemble import RandomForestClassifier


class CustomFeature(SetFeature):
    def extract(self, datasets):
        return [
            [nx.average_clustering(ds.data), np.mean([d for _, d in ds.data.degree()])]
            for ds in datasets
        ]


attacker = ShadowModellingAttack(
    FeatureBasedSetClassifier(
        features=CustomFeature(), classifier=RandomForestClassifier(),
    ),
    label="CustomGroundhog",
)

print("Training the attack...")
attacker.train(threat_model, num_samples=200)

print("Testing the attack...")
attack_summary = threat_model.test(attacker, num_samples=100)

metrics = attack_summary.get_metrics()
print("Results:\n", metrics.head())
