"""
Threat models specific to network datasets (tapas.datasets.network).

As for record-based data, a threat model is composed of three components:
 1. What the attacker aims to infer (e.g., membership, attribute).
 2. What the attacker knows about the generator (no-, black-, white-box).
 3. What the attacker knows about the training dataset.

Steps 1 and 2 are mostly identical to tabular data (except that some attacker
goals do not make sense, in particular, attribute inference).
Classes implemented in other parts of this module can thus be reused.

The following classes can be used transparently, without changes from tabular:
- .attacker_knowledge.AttackerKnowledgeOnGenerator (and children).
- .attacker_knowledge.ExactDataKnowledge
- .attacker_knowledge.LabelInferenceThreatModel
- .mia.TargetedMIA
- .mia.MIALabeller with replace_target = False.

The main class that cannot be ported transparently is AuxiliaryDataKnowledge.
More generally, the notion of "auxiliary dataset" cannot be ported easily for
network datasets. In this file, we define auxiliary knowledge for attackers
over networks. Such attacker knowledge can always be defined as a prior distribution
over the set of possible networks. We also recall which classes can be used as is.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..attacks import Attack
    from ..datasets import Dataset
    from ..generators import Generator
    from collections.abc import Iterable
    from typing import Callable

from .attacker_knowledge import AttackerKnowledgeWithLabel
from .mia import MIALabeller, TargetedMIA


# Cheeky lambda function instead of a class (to clean up at some point)
GraphTargetedMIA = lambda *args, **kwargs: TargetedMIA(*args, **kwargs, replace_target = False)
