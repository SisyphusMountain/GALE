//! Forward simulation of gene families under GALE's undated duplication-transfer-loss model
//! (`gpurec.simulate_families`, whose docstring states the model).
//!
//! One family: a first copy on a branch drawn from the origination probabilities; every copy on
//! branch e then undergoes one event, with probabilities proportional to 1 (speciation), D_e
//! (duplication), L_e (loss) and T_e (transfer). A speciation sends one copy to each child
//! branch, or on a leaf branch makes a sampled gene; a duplication leaves two copies on e; a
//! transfer leaves one copy on e and sends one to a branch h that is neither e nor an ancestor
//! of e, drawn with probability proportional to exp(w_h); a loss ends the copy. A family that
//! leaves no gene, or whose history exceeds the copy cap, is drawn again (and counted).
//!
//! The kept part of the history is written as the sampler's node list (`SampleNode`, read by
//! `gpurec.reconciliation.tree.tree_from_sampler_nodes`): the events GALE's sampler shows, with a
//! "loss" child for a speciation with a lost copy (kept child first) and for a transfer whose
//! copy left on the donor is lost (that loss first). A duplication with a lost copy, and a
//! transfer whose transferred copy is lost, are not shown, as in the sampler.

use crate::SampleNode;
use numpy::ndarray::{Array2, Array3};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

const NONE: usize = usize::MAX;
// Rejected receiver draws (a draw on the donor or one of its ancestors) before the receiver is
// drawn by a walk over every branch instead. Both draws have the same distribution: the cap
// changes no probability, it only bounds the loop when the ancestors hold most of the weight.
const RECEIVER_REJECTIONS: usize = 64;
// Attempts allowed for one family before the simulation stops with an error: past this many
// attempts that left no gene or went over the copy cap, the settings almost never give a family
// (for instance a cap too small for any history that reaches a leaf).
const MAX_ATTEMPTS: u64 = 1_000_000;

#[derive(Clone, Copy, PartialEq)]
enum Event {
    Speciation,
    Duplication,
    Loss,
    Transfer,
}

/// The species tree, the origination and receiver weights, and the rate rows.
struct Model<'a> {
    child1: &'a [i64],
    child2: &'a [i64],
    /// Pre-order position of each branch and one past the last branch below it: h is e or an
    /// ancestor of e exactly when `pre[h] <= pre[e] < end[h]`.
    pre: Vec<usize>,
    end: Vec<usize>,
    /// Running sums of exp(w - max w) over the branches, and of the origination probabilities.
    receiver_cdf: Vec<f64>,
    origin_cdf: Vec<f64>,
    /// [rows, branches, 3]: (D, L, T) of each branch; one row, or one per family.
    rates: &'a [f64],
    rows: usize,
}

/// One family: its node list, Newick text, gene names (in the order of the text), the species
/// of each gene, the events of the whole history (speciations on internal branches,
/// duplications, losses, transfers, sampled genes), and the attempts drawn again (no gene, over
/// the copy cap, some genes but fewer than the minimum).
type Family = (Vec<SampleNode>, String, Vec<String>, Vec<usize>, [u64; 5], u64, u64, u64);
/// A history: the event, branch and two children (`NONE` for none) of each copy.
type History = (Vec<Event>, Vec<usize>, Vec<[usize; 2]>);

fn draw_cdf(cdf: &[f64], rng: &mut StdRng) -> usize {
    let u = rng.gen::<f64>() * cdf[cdf.len() - 1];
    cdf.partition_point(|&c| c <= u).min(cdf.len() - 1)
}

impl Model<'_> {
    fn leaf(&self, e: usize) -> bool {
        self.child1[e] < 0
    }

    fn banned(&self, h: usize, donor: usize) -> bool {
        self.pre[h] <= self.pre[donor] && self.pre[donor] < self.end[h]
    }

    fn receiver(&self, donor: usize, rng: &mut StdRng) -> usize {
        for _ in 0..RECEIVER_REJECTIONS {
            let h = draw_cdf(&self.receiver_cdf, rng);
            if !self.banned(h, donor) {
                return h;
            }
        }
        let weight = |h: usize| {
            let below = if h == 0 { 0.0 } else { self.receiver_cdf[h - 1] };
            if self.banned(h, donor) { 0.0 } else { self.receiver_cdf[h] - below }
        };
        let total: f64 = (0..self.pre.len()).map(weight).sum();
        let mut u = rng.gen::<f64>() * total;
        let mut last = NONE;
        for h in 0..self.pre.len() {
            let w = weight(h);
            if w > 0.0 {
                last = h;
                u -= w;
                if u < 0.0 {
                    return h;
                }
            }
        }
        last
    }

    /// One attempt: the history as (event, branch, children) per copy, parents before
    /// children, or None past `max_copies` copies.
    fn history(&self, row: usize, max_copies: usize, rng: &mut StdRng) -> Option<History> {
        let s = self.pre.len();
        let (mut event, mut branch, mut kids) = (vec![Event::Loss], vec![draw_cdf(&self.origin_cdf, rng)], vec![[NONE; 2]]);
        let mut stack = vec![0usize];
        while let Some(v) = stack.pop() {
            let e = branch[v];
            let r = &self.rates[3 * (row * s + e)..3 * (row * s + e) + 3];
            let u = rng.gen::<f64>() * (1.0 + r[0] + r[1] + r[2]);
            let (kind, children) = if u < 1.0 {
                if self.leaf(e) {
                    (Event::Speciation, None)
                } else {
                    (Event::Speciation, Some((self.child1[e] as usize, self.child2[e] as usize)))
                }
            } else if u < 1.0 + r[0] {
                (Event::Duplication, Some((e, e)))
            } else if u < 1.0 + r[0] + r[1] {
                (Event::Loss, None)
            } else {
                (Event::Transfer, Some((e, self.receiver(e, rng)))) // donor copy first
            };
            event[v] = kind;
            if let Some((f, g)) = children {
                if branch.len() + 2 > max_copies {
                    return None;
                }
                let a = branch.len();
                event.extend([Event::Loss, Event::Loss]);
                branch.extend([f, g]);
                kids.extend([[NONE; 2], [NONE; 2]]);
                kids[v] = [a, a + 1];
                stack.extend([a + 1, a]);
            }
        }
        Some((event, branch, kids))
    }

    fn family(&self, labels: &[String], index: usize, seed: u64, max_copies: usize, min_genes: usize)
        -> Result<Family, String> {
        let mut rng = StdRng::seed_from_u64(mix(mix(seed) ^ index as u64));
        let row = if self.rows == 1 { 0 } else { index };
        let (mut no_gene, mut over_cap, mut too_few) = (0u64, 0u64, 0u64);
        loop {
            if no_gene + over_cap + too_few >= MAX_ATTEMPTS {
                return Err(format!("family {index}: {no_gene} attempts left no gene, {over_cap} went over \
                                    {max_copies} copies and {too_few} left fewer than {min_genes} genes, none kept: \
                                    the rates, the copy cap or the minimum give almost no family"));
            }
            let Some((event, branch, kids)) = self.history(row, max_copies, &mut rng) else {
                over_cap += 1;
                continue;
            };
            // children come after their parent: a backward pass sees them first
            let mut seen = vec![false; event.len()];
            for v in (0..event.len()).rev() {
                seen[v] = match kids[v] {
                    [NONE, _] => event[v] == Event::Speciation,
                    [a, b] => seen[a] || seen[b],
                };
            }
            if !seen[0] {
                no_gene += 1;
                continue;
            }
            let mut raw = [0u64; 5];
            for (v, kind) in event.iter().enumerate() {
                let k = match kind {
                    Event::Speciation if kids[v][0] == NONE => 4,
                    Event::Speciation => 0,
                    Event::Duplication => 1,
                    Event::Loss => 2,
                    Event::Transfer => 3,
                };
                raw[k] += 1;
            }
            // raw[4]: the sampled genes (speciations on leaf branches)
            if (raw[4] as usize) < min_genes {
                too_few += 1;
                continue;
            }
            let (nodes, genes) = shown(&event, &branch, &kids, &seen);
            let (newick, names) = newick(&nodes, &genes, labels);
            return Ok((nodes, newick, names, genes, raw, no_gene, over_cap, too_few));
        }
    }
}

/// The node list of the kept history, as the sampler writes it (module docstring), and the
/// species of each gene, genes numbered in the order of the text (first child first).
fn shown(event: &[Event], branch: &[usize], kids: &[[usize; 2]], seen: &[bool])
    -> (Vec<SampleNode>, Vec<usize>) {
    let (mut nodes, mut genes): (Vec<SampleNode>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut stack = vec![(0usize, NONE, 0usize)];
    while let Some((mut v, parent, slot)) = stack.pop() {
        // steps that stay on their branch are not shown
        loop {
            let [a, b] = kids[v];
            let hidden = a != NONE && seen[a] != seen[b]
                && (event[v] == Event::Duplication || (event[v] == Event::Transfer && seen[a]));
            if !hidden {
                break;
            }
            v = if seen[a] { a } else { b };
        }
        let o = nodes.len();
        if parent != NONE {
            nodes[parent].2[slot] = o as i64;
        }
        let [a, b] = kids[v];
        let kind = match event[v] {
            Event::Speciation if a == NONE => "leaf",
            Event::Speciation => "speciation",
            Event::Duplication => "duplication",
            Event::Transfer => "transfer",
            Event::Loss => unreachable!("a lost copy is never shown"),
        };
        let clade = if a == NONE { genes.push(branch[v]); genes.len() as i64 - 1 } else { -1 };
        nodes.push((kind, branch[v], [-1, -1], clade));
        if a == NONE {
            continue;
        }
        if seen[a] && seen[b] {
            stack.extend([(b, o, 1), (a, o, 0)]);
        } else {
            // speciation with a lost copy: kept child, then the loss; transfer whose donor copy
            // is lost: the loss (on the donor), then the transferred copy
            let (kept, lost) = if seen[a] { (a, b) } else { (b, a) };
            let (kept_slot, lost_slot) = if event[v] == Event::Transfer { (1, 0) } else { (0, 1) };
            nodes.push(("loss", branch[lost], [-1, -1], -1));
            nodes[o].2[lost_slot] = nodes.len() as i64 - 1;
            stack.push((kept, o, kept_slot));
        }
    }
    (nodes, genes)
}

/// The gene tree of a node list as Newick text, genes named `<species>_<k>`, k counting from 1
/// within each species in the order of the text; and the names, in that order.
fn newick(nodes: &[SampleNode], genes: &[usize], labels: &[String]) -> (String, Vec<String>) {
    let mut count = vec![0usize; labels.len()];
    let names: Vec<String> = genes.iter().map(|&s| {
        count[s] += 1;
        format!("{}_{}", labels[s], count[s])
    }).collect();
    enum Token { Node(usize), Text(char) }
    let (mut text, mut stack) = (String::new(), vec![Token::Node(0)]);
    while let Some(token) = stack.pop() {
        let mut i = match token {
            Token::Text(c) => { text.push(c); continue; }
            Token::Node(i) => i,
        };
        loop {
            let (kind, _, [a, b], clade) = nodes[i];
            if kind == "leaf" {
                text.push_str(&names[clade as usize]);
                break;
            }
            let (a, b) = (a as usize, b as usize);
            match (nodes[a].0, nodes[b].0) {
                (_, "loss") => i = a,
                ("loss", _) => i = b,
                _ => {
                    text.push('(');
                    stack.extend([Token::Text(')'), Token::Node(b), Token::Text(','), Token::Node(a)]);
                    break;
                }
            }
        }
    }
    text.push(';');
    (text, names)
}

/// SplitMix64's mixing function: spreads the seed and the family index over all 64 bits.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn cdf(weights: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut total = 0.0;
    weights.map(|w| { total += w; total }).collect()
}

/// `n_families` families that each leave at least one gene. Family i draws from its own random
/// generator, seeded by `seed` and i, so the output does not depend on `threads`.
/// Columns of the per-species event counts: AleRax's speciesEventCounts, the columns of
/// `gpurec.reconciliation.files.SPECIES_COUNT_HEADER`.
const COUNT_COLUMNS: usize = 9;

/// The code of a node kind in the flat arrays (`gpurec.simulate.EVENTS` names them in this order).
fn event_code(kind: &str) -> i8 {
    match kind {
        "leaf" => 0,
        "speciation" => 1,
        "duplication" => 2,
        "transfer" => 3,
        "loss" => 4,
        _ => unreachable!("the sampler's node kinds are these five"),
    }
}

/// A speciation, or a sampled gene (leaf): the counts of `gpurec.reconciliation.files`'s rule for
/// S, SL and Leaf. `prev` is the event before it on the same copy.
fn add_speciation(c: &mut [f64], sp: usize, prev: &str) {
    let row = sp * COUNT_COLUMNS;
    c[row] += 1.0;
    c[row + 4] = 1.0;
    c[row + 6] += 1.0;
    if prev == "S" || prev == "SL" {
        c[row + 7] += 1.0;
    }
}

/// The per-species event counts of one reconciled tree, row-major [species, COUNT_COLUMNS]: the
/// rules of `gpurec.reconciliation.files.species_event_counts`, read from the node list. Written
/// to match it exactly, and checked against it in tests/test_simulate.py.
fn event_counts_of(nodes: &[SampleNode], species_count: usize) -> Vec<f64> {
    let mut c = vec![0f64; species_count * COUNT_COLUMNS];
    let mut stack: Vec<(usize, &'static str)> = vec![(0, "S")];
    while let Some((mut i, mut prev)) = stack.pop() {
        loop {
            let (kind, sp, [a, b], _) = nodes[i];
            match kind {
                "speciation" if a >= 0 && nodes[b as usize].0 == "loss" => {
                    // a speciation with a lost copy: the kept child first, then the loss
                    add_speciation(&mut c, sp, prev);
                    c[nodes[b as usize].1 * COUNT_COLUMNS + 2] += 1.0;
                    prev = "SL";
                    i = a as usize;
                }
                "transfer" if a >= 0 && nodes[a as usize].0 == "loss" => {
                    // a transfer whose donor copy is lost
                    c[sp * COUNT_COLUMNS + 2] += 1.0;
                    c[sp * COUNT_COLUMNS + 3] += 1.0;
                    c[nodes[b as usize].1 * COUNT_COLUMNS + 8] += 1.0;
                    prev = "TL";
                    i = b as usize;
                }
                "leaf" => {
                    add_speciation(&mut c, sp, prev);
                    break;
                }
                "speciation" => {
                    add_speciation(&mut c, sp, prev);
                    stack.push((a as usize, "S"));
                    stack.push((b as usize, "S"));
                    break;
                }
                "duplication" => {
                    c[sp * COUNT_COLUMNS + 1] += 1.0;
                    stack.push((a as usize, "D"));
                    stack.push((b as usize, "D"));
                    break;
                }
                "transfer" => {
                    // the donor copy stays on the transfer's species, the recipient goes elsewhere
                    let (donor, recipient) = if nodes[a as usize].1 == sp { (a, b) } else { (b, a) };
                    c[sp * COUNT_COLUMNS + 3] += 1.0;
                    c[nodes[recipient as usize].1 * COUNT_COLUMNS + 8] += 1.0;
                    stack.push((donor as usize, prev));
                    stack.push((recipient as usize, "T"));
                    break;
                }
                _ => unreachable!("a lost copy is reached only from its speciation or transfer"),
            }
        }
    }
    // the origin: the species of the root event
    c[nodes[0].1 * COUNT_COLUMNS + 5] = 1.0;
    c
}

/// A batch of families as flat arrays, returned to Python as a dict (`gpurec.simulate`, whose
/// docstring gives each array's layout). `event_counts`: whether to count the per-species events
/// of every family (None in the result otherwise).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simulate_arrays<'py>(py: Python<'py>, child1: Vec<i64>, child2: Vec<i64>, labels: Vec<String>,
                            rates: Vec<f64>, receiver_log_weights: Vec<f64>, origination: Vec<f64>,
                            n_families: usize, seed: u64, max_copies: usize, min_genes: usize,
                            threads: usize, event_counts: bool)
    -> PyResult<Bound<'py, PyDict>> {
    let s = child1.len();
    let bad = |m: &str| Err(PyValueError::new_err(m.to_string()));
    if s < 3 || child2.len() != s || labels.len() != s || receiver_log_weights.len() != s || origination.len() != s {
        return bad("the species tree needs at least two leaves, and one label, receiver weight and origination probability per branch");
    }
    if rates.len() != 3 * s && rates.len() != 3 * s * n_families {
        return bad("rates: one (D, L, T) per branch, for every family or for each family");
    }
    if !rates.iter().all(|r| r.is_finite() && *r >= 0.0) || !receiver_log_weights.iter().all(|w| w.is_finite())
        || !origination.iter().all(|p| p.is_finite() && *p >= 0.0) || origination.iter().sum::<f64>() <= 0.0 {
        return bad("rates and origination probabilities must be finite and non-negative, receiver weights finite");
    }
    if max_copies < 1 || min_genes < 1 || threads < 1 {
        return bad("max_copies, min_genes and threads must be at least 1");
    }
    let parent = {
        let mut p = vec![NONE; s];
        for e in 0..s {
            for c in [child1[e], child2[e]] {
                if c >= 0 {
                    p[c as usize] = e;
                }
            }
        }
        p
    };
    let root = (0..s).find(|&e| parent[e] == NONE).unwrap();
    let (mut pre, mut end, mut stack, mut next) = (vec![0; s], vec![0; s], vec![(root, false)], 0);
    while let Some((e, done)) = stack.pop() {
        if done {
            end[e] = next;
            continue;
        }
        pre[e] = next;
        next += 1;
        stack.push((e, true));
        if child1[e] >= 0 {
            stack.extend([(child2[e] as usize, false), (child1[e] as usize, false)]);
        }
    }
    let top = receiver_log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let model = Model {
        child1: &child1, child2: &child2, pre, end,
        receiver_cdf: cdf(receiver_log_weights.iter().map(|w| (w - top).exp())),
        origin_cdf: cdf(origination.iter().cloned()),
        rates: &rates, rows: rates.len() / (3 * s),
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let results = py.allow_threads(|| pool.install(|| (0..n_families).into_par_iter()
        .map(|i| model.family(&labels, i, seed, max_copies, min_genes).map(|family| {
            let counts = event_counts.then(|| event_counts_of(&family.0, s));
            (family, counts)
        }))
        .collect::<Result<Vec<_>, _>>()))
        .map_err(PyValueError::new_err)?;

    let (mut node_offsets, mut gene_offsets) = (vec![0i64], vec![0i64]);
    let (mut parent, mut children, mut branch, mut event, mut gene) =
        (Vec::<i32>::new(), Vec::<i32>::new(), Vec::<i32>::new(), Vec::<i8>::new(), Vec::<i32>::new());
    let (mut gene_names, mut newick) = (Vec::<String>::new(), Vec::<String>::new());
    let (mut history, mut no_gene, mut over_cap, mut too_few) =
        (Vec::<i64>::new(), Vec::<i64>::new(), Vec::<i64>::new(), Vec::<i64>::new());
    let mut counts_all = Vec::<f64>::new();
    for (family, counts) in results {
        let (nodes, text, names, _genes, raw, ng, oc, tf) = family;
        // parents of the local nodes: the node whose child slot holds each node
        let mut local_parent = vec![-1i32; nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            for c in node.2 {
                if c >= 0 {
                    local_parent[c as usize] = i as i32;
                }
            }
        }
        for (i, node) in nodes.iter().enumerate() {
            parent.push(local_parent[i]);
            children.extend(node.2.iter().map(|&c| c as i32));
            branch.push(node.1 as i32);
            event.push(event_code(node.0));
            gene.push(node.3 as i32);
        }
        node_offsets.push(event.len() as i64);
        gene_offsets.push(gene_offsets[gene_offsets.len() - 1] + names.len() as i64);
        gene_names.extend(names);
        newick.push(text);
        history.extend(raw.iter().map(|&x| x as i64));
        no_gene.push(ng as i64);
        over_cap.push(oc as i64);
        too_few.push(tf as i64);
        if let Some(c) = counts {
            counts_all.extend(c);
        }
    }
    let families = n_families;
    let total_nodes = event.len();
    let dict = PyDict::new_bound(py);
    dict.set_item("node_offsets", node_offsets.into_pyarray_bound(py))?;
    dict.set_item("parent", parent.into_pyarray_bound(py))?;
    dict.set_item("children", Array2::from_shape_vec((total_nodes, 2), children)
        .map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py))?;
    dict.set_item("branch", branch.into_pyarray_bound(py))?;
    dict.set_item("event", event.into_pyarray_bound(py))?;
    dict.set_item("gene", gene.into_pyarray_bound(py))?;
    dict.set_item("gene_offsets", gene_offsets.into_pyarray_bound(py))?;
    dict.set_item("gene_names", gene_names)?;
    dict.set_item("newick", newick)?;
    dict.set_item("history", Array2::from_shape_vec((families, 5), history)
        .map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py))?;
    dict.set_item("attempts_without_gene", no_gene.into_pyarray_bound(py))?;
    dict.set_item("attempts_over_cap", over_cap.into_pyarray_bound(py))?;
    dict.set_item("attempts_too_few_genes", too_few.into_pyarray_bound(py))?;
    if event_counts {
        let shape = (families, s, COUNT_COLUMNS);
        dict.set_item("event_counts", Array3::from_shape_vec(shape, counts_all)
            .map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray_bound(py))?;
    } else {
        dict.set_item("event_counts", py.None())?;
    }
    Ok(dict)
}
