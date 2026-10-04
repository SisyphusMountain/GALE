from gpurec.io.species_tree import species_tree_from_arrays
from gpurec.io.text import format_number
from gpurec.reconciliation.files import (
    event_counts_text,
    recphyloxml_text,
    samples_newick_line,
    species_event_counts,
    species_event_counts_text,
    transfers_text,
)
from gpurec.reconciliation.tree import tree_from_sampler_nodes

A, B, AB, C, R = 0, 1, 2, 3, 4
SPECIES = species_tree_from_arrays(["A", "B", "AB", "C", "R"], [1.0] * 5, [-1, -1, 0, -1, 2], [-1, -1, 1, -1, 3])
HEADER = (
    "<recPhylo \n"
    '\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
    '\txsi:schemaLocation="http://www.recg.org ./recGeneTreeXML.xsd"\n'
    '\txmlns="http://www.recg.org">\n'
    "<spTree>\n<phylogeny>\n"
    "<clade>\n\t<name>R</name>\n"
    "\t<clade>\n\t\t<name>AB</name>\n"
    "\t\t<clade>\n\t\t\t<name>A</name>\n\t\t</clade>\n"
    "\t\t<clade>\n\t\t\t<name>B</name>\n\t\t</clade>\n"
    "\t</clade>\n"
    "\t<clade>\n\t\t<name>C</name>\n\t</clade>\n"
    "</clade>\n"
    "</phylogeny>\n</spTree>\n"
    '<recGeneTree>\n<phylogeny rooted="true">\n'
)
FOOTER = "</phylogeny>\n</recGeneTree>\n</recPhylo>\n"


def _speciation_tree():
    nodes = [
        ("speciation", R, [1, 2], 10),
        ("speciation", AB, [3, 4], 5),
        ("leaf", C, [-1, -1], 7),
        ("leaf", A, [-1, -1], 5),
        ("loss", B, [-1, -1], -1),
    ]
    return tree_from_sampler_nodes(nodes, {5: "a1", 7: "c1"})


def _transfer_tree():
    nodes = [("transfer", A, [1, 2], 9), ("leaf", C, [-1, -1], 4), ("leaf", A, [-1, -1], 3)]
    return tree_from_sampler_nodes(nodes, {3: "a1", 4: "c1"})


def test_format_number():
    assert [format_number(x) for x in (3, 3.0, 0.5, 1e-05, 0.1)] == ["3", "3", "0.5", "1e-05", "0.1"]


def test_recphyloxml_speciation_loss():
    gene = (
        "<clade>\n\t<name>NULL</name>\n"
        '\t<eventsRec>\n\t\t<speciation speciesLocation="R"/>\n\t</eventsRec>\n'
        "\t<clade>\n\t\t<name>NULL</name>\n"
        '\t\t<eventsRec>\n\t\t\t<speciation speciesLocation="AB"/>\n\t\t</eventsRec>\n'
        "\t\t<clade>\n\t\t\t<name>loss</name>\n"
        '\t\t\t<eventsRec>\n\t\t\t\t<loss speciesLocation="B"/>\n\t\t\t</eventsRec>\n'
        "\t\t</clade>\n"
        "\t\t<clade>\n\t\t\t<name>a1</name>\n"
        '\t\t\t<eventsRec>\n\t\t\t\t<leaf speciesLocation="A"/>\n\t\t\t</eventsRec>\n'
        "\t\t</clade>\n"
        "\t</clade>\n"
        "\t<clade>\n\t\t<name>c1</name>\n"
        '\t\t<eventsRec>\n\t\t\t<leaf speciesLocation="C"/>\n\t\t</eventsRec>\n'
        "\t</clade>\n"
        "</clade>\n"
    )
    assert recphyloxml_text(_speciation_tree(), SPECIES) == HEADER + gene + FOOTER


def test_recphyloxml_transfer_back_on_recipient_side():
    gene = (
        "<clade>\n\t<name>NULL</name>\n"
        '\t<eventsRec>\n\t\t<branchingOut speciesLocation="A"/>\n\t</eventsRec>\n'
        "\t<clade>\n\t\t<name>a1</name>\n"
        '\t\t<eventsRec>\n\t\t\t<leaf speciesLocation="A"/>\n\t\t</eventsRec>\n'
        "\t</clade>\n"
        "\t<clade>\n\t\t<name>c1</name>\n"
        '\t\t<eventsRec>\n\t\t\t<transferBack destinationSpecies="C"/>\n'
        '\t\t\t<leaf speciesLocation="C"/>\n\t\t</eventsRec>\n'
        "\t</clade>\n"
        "</clade>\n"
    )
    assert recphyloxml_text(_transfer_tree(), SPECIES) == HEADER + gene + FOOTER


def test_newick_event_and_transfer_files():
    tree = _speciation_tree()
    assert samples_newick_line(tree) == "(a1,c1)S;\n"
    assert event_counts_text(tree) == "S:1\nSL:1\nD:0\nDL:0\nT:0\nTL:0\nL:0\nLeaf:2\n"
    assert transfers_text(tree, SPECIES) == ""
    assert transfers_text(_transfer_tree(), SPECIES) == "A C 1\n"


def test_species_event_counts():
    header = (
        "species_label, speciations, duplications, losses, transfers, presence, origination, "
        "copies, singletons, transfers_to\n"
    )
    counts = species_event_counts(_speciation_tree(), SPECIES)
    assert species_event_counts_text(counts, SPECIES) == header + (
        "A, 1, 0, 0, 0, 1, 0, 1, 1, 0\n"
        "AB, 1, 0, 0, 0, 1, 0, 1, 1, 0\n"
        "B, 0, 0, 1, 0, 0, 0, 0, 0, 0\n"
        "C, 1, 0, 0, 0, 1, 0, 1, 1, 0\n"
        "R, 1, 0, 0, 0, 1, 1, 1, 1, 0\n"
    )
    counts = species_event_counts(_transfer_tree(), SPECIES)
    assert species_event_counts_text(counts, SPECIES) == header + (
        "A, 1, 0, 0, 1, 1, 1, 1, 1, 0\n"
        "C, 1, 0, 0, 0, 1, 0, 1, 0, 1\n"
    )


def test_recphyloxml_chain_of_transfer_loss_then_speciation_loss():
    # One gene node: transfer from C to AB with the donor copy lost (TL), then
    # speciation at AB with the copy in B lost (SL), then the leaf a1 in A.
    nodes = [
        ("transfer", C, [1, 2], 9),
        ("loss", C, [-1, -1], -1),
        ("speciation", AB, [3, 4], 9),
        ("leaf", A, [-1, -1], 9),
        ("loss", B, [-1, -1], -1),
    ]
    tree = tree_from_sampler_nodes(nodes, {9: "a1"})
    gene = (
        "<clade>\n\t<name>NULL</name>\n"
        '\t<eventsRec>\n\t\t<branchingOut speciesLocation="C"/>\n\t</eventsRec>\n'
        "\t<clade>\n\t\t<name>loss</name>\n"
        '\t\t<eventsRec>\n\t\t\t<loss speciesLocation="C"/>\n\t\t</eventsRec>\n'
        "\t</clade>\n"
        "\t<clade>\n\t\t<name>NULL</name>\n"
        '\t\t<eventsRec>\n\t\t\t<transferBack destinationSpecies="AB"/>\n'
        '\t\t\t<speciation speciesLocation="AB"/>\n\t\t</eventsRec>\n'
        "\t\t<clade>\n\t\t\t<name>loss</name>\n"
        '\t\t\t<eventsRec>\n\t\t\t\t<loss speciesLocation="B"/>\n\t\t\t</eventsRec>\n'
        "\t\t</clade>\n"
        "\t\t<clade>\n\t\t\t<name>a1</name>\n"
        '\t\t\t<eventsRec>\n\t\t\t\t<leaf speciesLocation="A"/>\n\t\t\t</eventsRec>\n'
        "\t\t</clade>\n"
        "\t</clade>\n"
        "</clade>\n"
    )
    assert recphyloxml_text(tree, SPECIES) == HEADER + gene + FOOTER
    assert event_counts_text(tree) == "S:0\nSL:1\nD:0\nDL:0\nT:0\nTL:1\nL:0\nLeaf:1\n"
    assert transfers_text(tree, SPECIES) == "C AB 1\n"
