from gpurec.reconciliation.tree import Event, tree_from_sampler_nodes

A, B, AB, C, R = 0, 1, 2, 3, 4


def test_speciation_with_loss_and_leaves():
    nodes = [
        ("speciation", R, [1, 2], 10),
        ("speciation", AB, [3, 4], 5),
        ("leaf", C, [-1, -1], 7),
        ("leaf", A, [-1, -1], 5),
        ("loss", B, [-1, -1], -1),
    ]
    tree = tree_from_sampler_nodes(nodes, {5: "a1", 7: "c1"})
    assert tree.root.events == (Event("S", R, -1, -1, "S"),)
    assert tree.root.left.events == (Event("SL", AB, A, B, "S"), Event("Leaf", A, -1, -1, "SL"))
    assert tree.root.left.gene == "a1"
    assert tree.root.right.events == (Event("Leaf", C, -1, -1, "S"),)
    assert tree.genes() == ["a1", "c1"]
    assert tree.origin == R
    assert [e.kind for e in tree.events()] == ["S", "SL", "Leaf", "Leaf"]


def test_transfer_children_are_donor_first():
    nodes = [
        ("transfer", A, [1, 2], 9),
        ("leaf", C, [-1, -1], 4),   # recipient side, listed first by the sampler
        ("leaf", A, [-1, -1], 3),   # donor side
    ]
    tree = tree_from_sampler_nodes(nodes, {3: "a1", 4: "c1"})
    assert tree.root.events == (Event("T", A, C, -1, "S"),)
    assert tree.root.left.gene == "a1" and tree.root.left.events[0].previous == "S"
    assert tree.root.right.gene == "c1" and tree.root.right.events[0].previous == "T"


def test_transfer_with_donor_loss():
    nodes = [("transfer", A, [1, 2], 9), ("loss", A, [-1, -1], -1), ("leaf", C, [-1, -1], 9)]
    tree = tree_from_sampler_nodes(nodes, {9: "c1"})
    assert tree.root.events == (Event("TL", A, C, -1, "S"), Event("Leaf", C, -1, -1, "TL"))


def test_single_leaf_tree():
    tree = tree_from_sampler_nodes([("leaf", A, [-1, -1], 0)], {0: "a1"})
    assert tree.root.events == (Event("Leaf", A, -1, -1, "S"),)
    assert tree.genes() == ["a1"]


def test_deep_caterpillar_does_not_hit_the_recursion_limit():
    depth = 3000
    nodes, clade_to_gene = [], {}
    for level in range(depth):
        base = len(nodes)
        nodes.append(("duplication", A, [base + 1, base + 2], 100000 + level))
        nodes.append(("leaf", A, [-1, -1], level))
        clade_to_gene[level] = f"g{level}"
    nodes.append(("leaf", A, [-1, -1], depth))
    clade_to_gene[depth] = f"g{depth}"
    tree = tree_from_sampler_nodes(nodes, clade_to_gene)
    assert len(tree.genes()) == depth + 1
