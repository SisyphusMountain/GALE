"""Probabilities of small named gene trees obtained by adding up histories, with no recursion.

TEST-ONLY reference for tests/test_duplication_split_factor.py. On the species tree ((X,Y),Z), with
fixed duplication, transfer and loss rates (speciation rate 1) and a family that starts on a
uniformly chosen branch, ``enumerate_histories`` lists every history with at most a given number of
gene copies and adds the products of their event probabilities, grouped by the gene tree they leave
(gene names attached later, uniformly at random within each species). That sum approaches the true
probability of each gene tree from below as the number of copies grows.

``Recursion`` is the clade recursion of the undated model on the same tree, written in plain
probabilities, with the number of times the term of a duplication whose two copies carry the two
child clades of a split counts (``split_count``) left open: 2 is the correct count (the two copies
can carry the two clades in two ways), 1 is the count of ALE's undated likelihood and AleRax 1.4.0.
The term of "duplication, then one copy lost" always counts twice.

A named tree is a gene name such as "X1" (species X) or a frozenset of two named trees. Ported from
the check of the factor 2 written for the paper (enumeration of histories and direct simulation).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

# Branches: 0, 1, 2 lead to the leaves X, Y, Z; 3 is the ancestor of X and Y; 4 is the root.
SPECIES_NEWICK = "((X,Y),Z);"
S = 5
CHILD = {3: (0, 1), 4: (3, 2)}
PARENT = {0: 3, 1: 3, 3: 4, 2: 4}
LEAF = {0: "X", 1: "Y", 2: "Z"}
# Duplication, transfer and loss rates; the speciation rate is 1.
DUPLICATION, TRANSFER, LOSS = 0.30, 0.22, 0.38


def _ancestors(branch):
    out = []
    while branch in PARENT:
        branch = PARENT[branch]
        out.append(branch)
    return out


# Branches a transfer from each branch may reach: neither the donor nor one of its ancestors.
RECEIVERS = {e: [h for h in range(S) if h != e and h not in _ancestors(e)] for e in range(S)}
_TOTAL = 1.0 + DUPLICATION + TRANSFER + LOSS
P_S, P_D, P_T, P_L = 1.0 / _TOTAL, DUPLICATION / _TOTAL, TRANSFER / _TOTAL, LOSS / _TOTAL
# The pairs of branches the two copies of an event go to, with the event's probability.
_PAIRS = {e: ([(CHILD[e][0], CHILD[e][1], P_S)] if e in CHILD else [])
             + [(e, e, P_D)] + [(e, h, P_T / len(RECEIVERS[e])) for h in RECEIVERS[e]] for e in range(S)}


def show(tree):
    return tree if isinstance(tree, str) else "(%s,%s)" % tuple(sorted(show(t) for t in tree))


def newick(tree):
    """Newick text with gene names ``X_1`` (the species before the first ``_``, as GALE reads it)."""
    if isinstance(tree, str):
        return "%s_%s" % (tree[0], tree[1:])
    a, b = sorted(tree, key=show)
    return "(%s,%s)" % (newick(a), newick(b))


def name_count(tree):
    """Number of ways of naming the genes of each species: the product of their factorials."""
    counts = Counter(ch for ch in show(tree) if ch in "XYZ")
    return math.prod(math.factorial(k) for k in counts.values())


def _orderings(tree):
    if isinstance(tree, str):
        return [tree[0]]
    a, b = tuple(tree)
    return ([(x, y) for x in _orderings(a) for y in _orderings(b)]
            + [(y, x) for x in _orderings(a) for y in _orderings(b)])


def extinction():
    """Probability that a copy on each branch leaves no sampled gene (fixed point, from zero)."""
    E = np.zeros(S)
    for _ in range(100000):
        Ebar = np.array([P_T / len(RECEIVERS[e]) * sum(E[h] for h in RECEIVERS[e]) for e in range(S)])
        new = P_D * E * E + E * Ebar + P_L
        for e, (f, g) in CHILD.items():
            new[e] += P_S * E[f] * E[g]
        if np.max(np.abs(new - E)) < 1e-16:
            return new
        E = new
    raise RuntimeError("extinction probabilities did not converge")


def _bar(values):
    return np.array([P_T / len(RECEIVERS[e]) * sum(values[h] for h in RECEIVERS[e]) for e in range(S)])


class Recursion:
    """The clade recursion; ``split_count`` is how many times the duplication term of a split counts."""

    def __init__(self, split_count):
        self.split_count = split_count
        self.cache = {}
        E = extinction()
        Ebar = _bar(E)
        M = np.zeros((S, S))
        for e in range(S):
            M[e, e] += 2.0 * P_D * E[e] + Ebar[e]
            for h in RECEIVERS[e]:
                M[e, h] += E[e] * P_T / len(RECEIVERS[e])
        for e, (f, g) in CHILD.items():
            M[e, f] += P_S * E[g]
            M[e, g] += P_S * E[f]
        self.inverse = np.linalg.inv(np.eye(S) - M)

    def _row(self, tree):
        """Per branch: name_count(tree) times the probability that one copy there yields tree."""
        if tree not in self.cache:
            b = np.zeros(S)
            if isinstance(tree, str):
                b[[k for k, s in LEAF.items() if s == tree[0]][0]] = P_S
            else:
                A, B = tuple(tree)
                PA, PB = self._row(A), self._row(B)
                b = self.split_count * P_D * PA * PB + PA * _bar(PB) + PB * _bar(PA)
                for e, (f, g) in CHILD.items():
                    b[e] += P_S * (PA[f] * PB[g] + PB[f] * PA[g])
            self.cache[tree] = self.inverse @ b
        return self.cache[tree]

    def probability(self, tree):
        """Probability of observing the rooted named tree, the family starting on a uniform branch."""
        return float(np.sum(self._row(tree))) / S / name_count(tree)


def _join(a, b):
    return b if a is None else (a if b is None else (a, b))


def _leaf_count(t):
    return 0 if t is None else (1 if isinstance(t, str) else _leaf_count(t[0]) + _leaf_count(t[1]))


def enumerate_histories(max_copies, max_genes):
    """``G[m][e][t]``: total probability of the histories from one copy on branch e that hold exactly
    m copies and leave the ordered species tree t (None: no sampled gene), t with at most max_genes
    genes."""
    G = [None, [defaultdict(float) for _ in range(S)]]
    for e in range(S):
        G[1][e][None] = P_L
        if e in LEAF:
            G[1][e][LEAF[e]] = P_S
    for m in range(2, max_copies + 1):
        G.append([defaultdict(float) for _ in range(S)])
        for e in range(S):
            for d1, d2, w in _PAIRS[e]:
                for m1 in range(1, m - 1):
                    for t1, q1 in G[m1][d1].items():
                        for t2, q2 in G[m - 1 - m1][d2].items():
                            t = _join(t1, t2)
                            if _leaf_count(t) <= max_genes:
                                G[m][e][t] += w * q1 * q2
    return G


def enumerated_probability(G, tree):
    """Probability of the rooted named tree from the enumerated histories (a lower bound)."""
    return sum(G[m][e].get(t, 0.0) for e in range(S) for m in range(1, len(G))
               for t in _orderings(tree)) / S / name_count(tree)


def rootings(tree):
    """The rooted trees of an unrooted tree with two or three leaves (GALE reads gene trees as
    unrooted and gives each rooting the same probability)."""
    if isinstance(tree, str):
        return [tree]
    leaves = sorted(show(tree).replace("(", "").replace(")", "").split(","))
    if len(leaves) == 2:
        return [tree]
    if len(leaves) != 3:
        raise ValueError(f"rootings of {show(tree)}: only two or three leaves are handled")
    a, b, c = leaves
    return [frozenset({frozenset({a, b}), c}), frozenset({frozenset({a, c}), b}),
            frozenset({frozenset({b, c}), a})]


def log_likelihood_nats(probability_of_rooted, tree, survival):
    """GALE's per-family log-likelihood of the unrooted named tree: the mean probability of its
    rootings, times the number of namings, over the probability that the family survives."""
    roots = rootings(tree)
    mean = sum(probability_of_rooted(r) for r in roots) / len(roots)
    return math.log(name_count(tree) * mean / survival)
