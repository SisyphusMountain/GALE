"""Readers for AleRax's input files: families file, gene-to-species mapping, fraction missing.

The rules follow AleRax 1.4.0 (GeneRaxCore ``FamiliesFileParser`` and ``GeneSpeciesMapping``):
all whitespace is removed from every line of a families file, ``#`` starts a comment, lines
before ``[FAMILIES]`` are ignored, and paths are used as written (relative to the current
directory, not to the families file).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Keys AleRax accepts that only concern sequence likelihoods, which gpurec does not compute.
_IGNORED_KEYS = frozenset({"alignment", "likelihoods", "subst_model"})
_TREE_KEYS = frozenset({"gene_tree", "starting_gene_tree"})


@dataclass(frozen=True)
class FamilyEntry:
    name: str
    gene_trees: Path
    mapping: Path | None


def _clean(line: str) -> str:
    return "".join(line.split("#", 1)[0].split())


def read_families_file(path: Path) -> list[FamilyEntry]:
    entries: list[FamilyEntry] = []
    seen: set[str] = set()
    current: dict | None = None
    in_families = False

    def close(record: dict | None) -> None:
        if record is None:
            return
        if record["gene_trees"] is None:
            raise ValueError(f"{path}: family {record['name']!r} has no gene tree (gene_tree=...)")
        entries.append(FamilyEntry(record["name"], record["gene_trees"], record["mapping"]))

    for number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = _clean(raw)
        if not line:
            continue
        if not in_families:
            in_families = line == "[FAMILIES]"
            continue
        if line.startswith("-"):
            close(current)
            name = line[1:]
            if name in seen:
                raise ValueError(f"{path}, line {number}: family {name!r} appears twice")
            seen.add(name)
            current = {"name": name, "gene_trees": None, "mapping": None}
            continue
        if current is None or "=" not in line:
            raise ValueError(f"{path}, line {number}: expected '- name' or 'key = value', got {raw.strip()!r}")
        key, value = line.split("=", 1)
        if not value:
            raise ValueError(f"{path}, line {number}: key {key!r} has no value")
        if key in _TREE_KEYS:
            current["gene_trees"] = Path(value)
        elif key == "mapping":
            current["mapping"] = Path(value)
        elif key not in _IGNORED_KEYS:
            raise ValueError(f"{path}, line {number}: unknown key {key!r}")
    close(current)
    if not in_families:
        raise ValueError(f"{path}: no [FAMILIES] line")
    return entries


def read_mapping_file(path: Path) -> dict[str, str]:
    numbered = [
        (number, line) for number, line in enumerate(Path(path).read_text().splitlines(), start=1) if line.strip()
    ]
    mapping: dict[str, str] = {}
    if numbered and ":" in numbered[0][1]:
        for number, line in numbered:
            if ":" not in line:
                raise ValueError(f"{path}, line {number}: expected 'species:gene1;gene2;...', got {line.strip()!r}")
            species, genes = line.strip().split(":", 1)
            for gene in genes.split(";"):
                if gene.strip():
                    mapping[gene.strip()] = species.strip()
    else:
        for number, line in numbered:
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{path}, line {number}: expected 'gene species', got {line.strip()!r}")
            mapping[fields[0]] = fields[1]
    return mapping


def read_fraction_missing_file(path: Path) -> dict[str, float]:
    tokens = Path(path).read_text().split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"{path}: expected pairs 'species fraction'")
    return {tokens[i]: float(tokens[i + 1]) for i in range(0, len(tokens), 2)}
