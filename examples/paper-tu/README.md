This folder contains a lot of notebooks. Here is how they should be understood.

## Applying simple attacks on a `TUDataset`: notebooks 1, 2, 3.

This applies simple targeted attacks on a `TUDataset`, i.e. a dataset where user records are individual graphs. The first notebook tries to attack the `Raw` generator and is successful (although w/o 100% accuracy). The second and third notebook attack a very simple generator (Barabasi-Albert with unnoised parameters estimated from the data). Surprisingly, the attacks were found to be more successful than random, but probably because the target was an "outlier" graph.

These notebooks overall show that a simple attack (treating the graph kernel as a feature vector) can work pretty well on a `TUDataset` of small size, especially for graphs that are quite different from others. The work is very preliminary though.

This work was put on hold because more realistic generators are needed before spending a lot of time evaluating privacy (an expensive process). Furthermore, the attack we currently use is very simple and can certainly be improved.


## Trying untargeted attacks: notebooks 4, 5, 6, 7

Inspired by encouraging results for tabular data, we now look at a different class of attacks: _untargeted_ attacks, where the classifier learns a function of (1) the synthetic `TUDataset` and (2) a record to predict membership of that record in the training dataset. In notebook 4, we try a trivial extension of the attack in notebooks 1-3, where we compute a feature vector for the synthetic data, one for the input record, concatenate them and feed them to a classifier. Early experiments on the `Raw` generator show that it does not work very well.

In notebook 5, we propose a new, more principled attack, where we compute the kernel between the input record and each synthetic graph and feed this to a classifier. Unsurprisingly, this achieves perfect testing accuracy on the `Raw` generator (since there is a 1 in the vector iff the target graph is in the data). We then apply this attack to one of the safe generators (the one from notebook 3) in notebook 6, and find that it works somewhat better than random, although the accuracy is very low. This is most likely because (1) the attack is very simple, and (2) the generator is safe (presumably). Notebook 7 replicates this analysis but for a simplistic generator that modifies individual graphs. The attack is not very successful overall, whichh is slightly worrying, as this generator seems easy to attack.

Similarly to above, this work was put on hold while waiting for more realistic generators. Untargeted attacks seem promising for future privacy analyses, given the cost of running a generator.


## Different Threat Model -- Node-based attacks: notebook 8

As a result of some generator implementations, we have come to realise that a lot of generators go from graph to graph (rather than datasets of graphs). Furthermore, assuming user records to be nodes rather than edges is more realistic from a privacy perspective (imho). The key challenge is thus to define a plausible threat model in terms of attacker knowledge. We review existing literature on the topic of graph deanonymisation in an Overleaf, and implement a simple solution on the `raw` generator. The results we obtain are underwhelming, but this is probably mostly because the attack is too simple.



