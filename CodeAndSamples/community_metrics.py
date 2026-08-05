#!/usr/bin/env python3
"""
community_metrics.py

Reproduces the network- and community-structure metrics reported in Table 2 of
"Citation Mapping of Predatory and Lower-Quality Publications in Nursing Literature."

Reads the incoming-citation node and edge CSVs produced by the retrieval pipeline
(run_citation_maps_local_cap200_*.py) and computes, deterministically, every value
in Table 2. Community detection uses the Louvain method (Blondel et al., 2008) with a
FIXED random seed (42) so the community count, modularity, and community-size figures
reproduce exactly on every run.

Node CSV columns : node, type, doi, depth   (depth 0 = seed article)
Edge CSV columns : source, target           (directed: citer -> cited)

Community detection and clustering are computed on the UNDIRECTED graph, on the
connected nodes only (degree >= 1), matching the manuscript.

Usage:
    python community_metrics.py --nodes SetA_nodes_attributes_IN.csv --edges SetA_edges_IN.csv --label Predatory
    python community_metrics.py --nodes SetB_nodes_attributes_IN.csv --edges SetB_edges_IN.csv --label INANE

Dependencies: networkx, python-louvain  (pip install networkx python-louvain)
"""
import argparse, csv, collections
import networkx as nx
import community.community_louvain as louvain

SEED = 42


def load(nodes_csv, edges_csv):
    depth = {}
    with open(nodes_csv, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            depth[r['node']] = int(r['depth'])
    edges = []
    with open(edges_csv, newline='', encoding='utf-8-sig') as f:
        rd = csv.reader(f); next(rd)          # skip header
        for row in rd:
            if len(row) >= 2 and row[0] and row[1]:
                edges.append((row[0], row[1]))
    return depth, edges


def subgraph_counts(depth, edges, max_depth):
    """Node count and directed edge count (unique citer->cited pairs, self-loops
    excluded) among nodes at or above the given hop depth. Directed pairs match the
    retrieval pipeline's DiGraph convention. NOTE: these are recomputed from the
    archived edge CSV and may differ from Table 2's originally reported edge counts by
    a few tenths of a percent, because the manuscript's counts were taken from the
    pipeline's live edge tally during retrieval rather than recomputed from the CSV."""
    keep = {n for n, d in depth.items() if d <= max_depth}
    ec = set()
    for s, t in edges:
        if s in keep and t in keep and s != t:
            ec.add((s, t))
    return len(keep), len(ec)


def analyze(label, nodes_csv, edges_csv):
    depth, edges = load(nodes_csv, edges_csv)
    total_nodes = len(depth)

    # ---- per-depth node / edge counts ----
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for d in (1, 2, 3):
        n, e = subgraph_counts(depth, edges, d)
        print(f"  {d}-hop nodes: {n:>7,}    {d}-hop edges: {e:>8,}")

    # ---- avg. citations received per article (one-hop citers / seeds) ----
    seeds = sum(1 for v in depth.values() if v == 0)
    onehop = sum(1 for v in depth.values() if v == 1)
    print(f"  Seeds: {seeds}   One-hop citers: {onehop}   "
          f"Avg. citations received per article: {onehop/seeds:.2f}")

    # ---- build UNDIRECTED graph from the EDGE LIST ----
    # Construct the graph edges-first (nodes are added in order of first appearance in
    # the edge list). Isolated nodes carry no edges and are therefore excluded, matching
    # the manuscript's "connected nodes only" community analysis. NOTE: this construction
    # order is significant -- the Louvain method visits nodes in insertion order, so an
    # edges-first build is required to reproduce the exact community counts in Table 2.
    Gc = nx.Graph()
    for s, t in edges:
        if s != t:
            Gc.add_edge(s, t)
    connected = Gc.number_of_nodes()
    isolates = total_nodes - connected
    print(f"  Total nodes: {total_nodes:,}   Connected (degree>=1): {connected:,}   "
          f"Isolates: {isolates:,}")

    # ---- Louvain community detection (fixed seed) ----
    part = louvain.best_partition(Gc, random_state=SEED)
    sizes = collections.Counter(part.values())
    n_comm = len(sizes)
    modularity = louvain.modularity(part, Gc)
    mean_size = connected / n_comm
    per_1000 = n_comm / connected * 1000
    ordered = sorted(sizes.values(), reverse=True)
    top3 = ordered[:3]
    top10_sum = sum(ordered[:10])
    top10_pct = top10_sum / connected * 100

    # ---- clustering & largest connected component ----
    avg_clustering = nx.average_clustering(Gc)
    comps = sorted((len(c) for c in nx.connected_components(Gc)), reverse=True)
    lcc = comps[0] if comps else 0
    lcc_pct = lcc / connected * 100

    print(f"  Communities detected**: {n_comm}")
    print(f"  Communities per 1,000 nodes**: {per_1000:.2f}")
    print(f"  Mean community size (nodes)**: {mean_size:.0f}")
    print(f"  Modularity**: {modularity:.3f}")
    print(f"  Avg. clustering coefficient**: {avg_clustering:.3f}")
    print(f"  Largest connected component**: {lcc:,} / {connected:,} ({lcc_pct:.1f}%)")
    print(f"  Top 3 community sizes**: {' / '.join(f'{x:,}' for x in top3)}")
    print(f"  Top 10 communities (% of network)**: {top10_pct:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reproduce Table 2 community metrics (Louvain, seed 42).")
    ap.add_argument("--nodes", required=True, help="node attributes CSV (node,type,doi,depth)")
    ap.add_argument("--edges", required=True, help="edges CSV (source,target)")
    ap.add_argument("--label", default="Network", help="label for output (e.g., Predatory / INANE)")
    a = ap.parse_args()
    analyze(a.label, a.nodes, a.edges)
