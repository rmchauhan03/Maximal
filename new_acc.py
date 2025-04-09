import pynauty
import networkx as nx
from collections import defaultdict
import time
import pickle
import os
from math import factorial
import json
from functools import lru_cache
import numpy as np
from multiprocessing import Process, cpu_count


THREAD_COUNT = cpu_count()

def nx_to_nauty(G):
    """Convert a NetworkX graph to a pynauty graph."""
    n = G.number_of_nodes()
    adjacency_dict = {}
    
    for v in range(n):
        neighbors = list(G.neighbors(v))
        if neighbors:  # Only add to adjacency_dict if the vertex has neighbors
            adjacency_dict[v] = set(neighbors)
    
    return pynauty.Graph(number_of_vertices=n, directed=False, adjacency_dict=adjacency_dict)

def nauty_to_nx(g_nauty):
    """Convert a pynauty graph to a NetworkX graph."""
    n = g_nauty.number_of_vertices
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for v in range(n):
        if v in g_nauty.adjacency_dict:
            for w in g_nauty.adjacency_dict[v]:
                if v < w:  # Add each edge only once
                    G.add_edge(v, w)
    
    return G

@lru_cache(maxsize=14)
def all_one_factors(n):
    """
    Generate a numpy array containing all one-factors of a complete graph on n vertices.
    
    Each row is a one-factor, represented as a permutation where the value at index i
    indicates the vertex that vertex i is paired with.
    
    Parameters:
    -----------
    n : int
        Number of vertices in the complete graph. Must be even.
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (m, n) where m is the number of one-factors.
    
    Notes:
    ------
    - A one-factor (or perfect matching) is a subset of edges where each vertex 
      is connected to exactly one other vertex.
    - For a complete graph on n vertices (where n is even), there are (n-1)!!
      different one-factors, where (n-1)!! is the double factorial of (n-1).
    
    Examples:
    ---------
    >>> all_one_factors(4)
    array([[1, 0, 3, 2],
           [2, 3, 0, 1],
           [3, 2, 1, 0]])
    
    >>> len(all_one_factors(6))
    15  # (6-1)!! = 5!! = 5 * 3 * 1 = 15
    """
    if n % 2 != 0:
        raise ValueError("n must be even for a one-factor to exist")
    
    all_factors = []
    
    def backtrack(factor, unmatched):
        """Recursively build all possible one-factors"""
        if not unmatched:
            # All vertices have been matched
            all_factors.append(factor.copy())
            return
        
        # Pick the first unmatched vertex
        v = unmatched[0]
        
        # Try matching it with each of the remaining unmatched vertices
        for u in unmatched[1:]:
            # Match v with u and u with v
            factor[v] = u
            factor[u] = v
            
            # Recursively match the remaining vertices
            new_unmatched = [x for x in unmatched if x != v and x != u]
            backtrack(factor, new_unmatched)
    
    # Initialize factor array
    factor = np.zeros(n, dtype=int)
    unmatched = list(range(n))
    
    backtrack(factor, unmatched)
    
    return np.array(all_factors)

def prune_edges(AllFac, edges):
    # Get all rows where the specified edge is not present
    for edge in edges:
        mask = AllFac[:, edge[0]] != edge[1]
        AllFac = AllFac[mask]
    return AllFac


def GenerateOneFactor(H):
    return prune_edges(all_one_factors(H.number_of_nodes()), H.edges)

class GraphRegistry:
    """
    Class to maintain a registry of representative graphs for each isomorphism class.
    """
    def __init__(self):
        self.graph_dict = {}  # certificate -> (G, nauty_graph, aut_size)

    def add_graph_cert(self, g_nauty, cert):
        if cert not in self.graph_dict:
            aut = pynauty.autgrp(g_nauty)
            self.graph_dict[cert] = (g_nauty, aut[1])
        return cert
    
    def merge_graph(self, other):
        for item in other.graph_dict:
            if item in self.graph_dict:
                continue
            self.graph_dict[item] = other.graph_dict[item]

    
    def add_graph(self, G):
        """Add a graph to the registry if it's a new isomorphism class."""
        g_nauty = nx_to_nauty(G)
        cert = pynauty.certificate(g_nauty)
        
        return self.add_graph_cert(g_nauty, cert)
    
    def get_graph(self, cert):
        if cert in self.graph_dict:
            return nauty_to_nx(self.graph_dict[cert][0])
        return None
    
    def get_nauty_graph(self, cert):
        """Get the pynauty graph for a certificate."""
        if cert in self.graph_dict:
            return self.graph_dict[cert][0]
        return None
    
    def get_aut_size(self, cert):
        """Get the automorphism group size for a certificate."""
        if cert in self.graph_dict:
            return self.graph_dict[cert][1]
        return None
    
    def save_to_file(self, filename):
        """Save the registry to a file."""
        # We can't directly pickle pynauty graphs, so we'll save the NetworkX graphs
        save_dict = {
            'graphs': {cert: (g_nauty, aut_size) for cert, ( g_nauty, aut_size) in self.graph_dict.items()}
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load_from_file(self, filename):
        """Load the registry from a file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                save_dict = pickle.load(f)
                
            # Reconstruct the pynauty graphs
            if 'graphs' in save_dict:
                for cert, ( g_nauty, aut_size) in save_dict['graphs'].items():
                    self.graph_dict[cert] = (g_nauty, aut_size)
                
            else:
                # Old format - just graphs
                for cert, (g_nauty, aut_size) in save_dict.items():
                    self.graph_dict[cert] = (g_nauty, aut_size)
            
            return True
        return False

def process_class(c, T, cert_list, prev_registry, LF_prev):

    registry = GraphRegistry()
    accumulators = defaultdict(float)  # Using Python's built-in float type

    _i = 0
    while _i < len(cert_list):
        cert_H = cert_list[_i]
        _i += 1
            
        # Get the representative graph H from the previous registry
        H = prev_registry.get_graph(cert_H)
        H_nauty = prev_registry.get_nauty_graph(cert_H)
        if H is None:
            continue
            
        # Get the precomputed automorphism group size
        aut_H_size = prev_registry.get_aut_size(cert_H)
        
        # The one-factorization count for H
        LF_H = LF_prev[cert_H]

        one_factors = GenerateOneFactor(H)
        
        if len(one_factors) == 0:
            continue            
        for F in one_factors:
            if len(F) == 0:
                continue
            
            g_nauty = pynauty.Graph(H_nauty.number_of_vertices, adjacency_dict=H_nauty.adjacency_dict.copy(), vertex_coloring= H_nauty.vertex_coloring.copy())
            for i in range(len(F)):
                if F[i] > i:
                    g_nauty.connect_vertex(i, F[i])
            cert_G = pynauty.certificate(g_nauty)
            
            # Now officially add the graph to the registry
            cert_G = registry.add_graph_cert(g_nauty, cert_G)
            
            # Get the automorphism group size of G
            aut_G_size = registry.get_aut_size(cert_G)
            
            # Calculate the contribution according to equation (5) in the paper:
            # The ratio |Aut(H ∪ F)|/|Aut(H)| gives how many times this graph is visited
            ratio = aut_G_size / aut_H_size  # Standard Python float division
            increment = ratio * LF_H
            
            # Update the accumulator
            accumulators[cert_G] += increment
    with open(f'{c}.pkl', 'wb') as file:
        pickle.dump((accumulators, registry), file)

def forward_accumulation(k, max_nodes=None, use_caching=True, cache_file=None, verbose=False):
    """
    Implement the forward accumulation approach to compute LF(G) for all
    k-regular one-factorizable graphs G.
    
    Args:
        k: The regularity of the graphs to consider
        max_nodes: Maximum number of nodes to consider (for limiting search)
        use_caching: Whether to use caching to speed up computations
        cache_file: File pattern to use for caching results
        verbose: Whether to print verbose progress information
    
    Returns:
        A dictionary mapping isomorphism classes (certificates) to their LF values,
        and a GraphRegistry containing representative graphs for each isomorphism class
    """
    print(f"Starting forward accumulation for k={k}")
    start_time = time.time()
    
    registry = GraphRegistry()
    
    # Check if we can load results from cache
    if use_caching and cache_file:
        cache_filename = f"{cache_file}_k{max_nodes}_{k}.pkl"
        if os.path.exists(cache_filename):
            print(f"Loading results from cache file: {cache_filename}")
            try:
                with open(cache_filename, 'rb') as f:
                    cached_results = pickle.load(f)
                    registry.load_from_file(f"{cache_file}_registry_k{max_nodes}_{k}.pkl")
                    print(f"Loaded {len(cached_results)} isomorphism classes from cache")
                    return cached_results, registry
            except Exception as e:
                print(f"Error loading from cache: {e}")
    
    if k < 1:
        return {}, registry
    
    # Base case: 1-regular graphs
    if k == 1:
        results = {}
        
        # For different numbers of vertices (2, 4, 6, ...)
        for n_pairs in range(1, (max_nodes // 2) + 1 if max_nodes else 4):
            n_vertices = 2 * n_pairs
            
            # Create a graph with n_pairs disjoint edges
            G = nx.Graph()
            G.add_nodes_from(range(n_vertices))
            for i in range(n_pairs):
                G.add_edge(2*i, 2*i + 1)
            
            # Add to registry and get certificate
            cert = registry.add_graph(G)
            
            # For 1-regular graphs, there's exactly 1 one-factorization
            results[cert] = 1
            
            if verbose:
                print(f"Added 1-regular graph with {n_vertices} vertices")
            
        # Save to cache if enabled
        if use_caching and cache_file:
            with open(f"{cache_file}_k{max_nodes}_{k}.pkl", 'wb') as f:
                pickle.dump(results, f)
            registry.save_to_file(f"{cache_file}_registry_k{max_nodes}_{k}.pkl")
            
        print(f"Completed k={k} in {time.time() - start_time:.2f} seconds")
        return results, registry
    
    # Recursively compute LF values for (k-1)-regular graphs
    LF_prev, prev_registry = forward_accumulation(k - 1, max_nodes, use_caching, cache_file, verbose)
    
    # Initialize accumulators for k-regular graphs
    accumulators = defaultdict(float)  # Using Python's built-in float type
    
    # Group graphs by vertex count for better processing
    vertex_to_certs = defaultdict(list)
    for cert_H in LF_prev.keys():
        H = prev_registry.get_graph(cert_H)
        if H:
            vertex_to_certs[H.number_of_nodes()].append(cert_H)
    
    print(f"Processing {len(LF_prev)} isomorphism classes of {k-1}-regular graphs")
        
    # Process smaller graphs first
    for n_vertices in sorted(vertex_to_certs.keys()):
        cert_list = vertex_to_certs[n_vertices]
        print(f"Processing {len(cert_list)} {k-1}-regular graphs with {n_vertices} vertices")
        
        # Process each isomorphism class
        threads = []
        cl = {i : [] for i in range(THREAD_COUNT)}
        for ii in range(len(cert_list)):
            cl[ii % THREAD_COUNT].append(cert_list[ii])
        del cert_list
        del vertex_to_certs[n_vertices]
        for i in range(THREAD_COUNT):
            if len(cl[i]) == 0:
                continue
            threads.append(Process(target=process_class, args=(i, THREAD_COUNT, cl[i], prev_registry, LF_prev)))
            threads[-1].start()
        
        for i in range(len(threads)):
            threads[i].join()


        for i in range(THREAD_COUNT):
            if os.path.exists(f"{i}.pkl"):
                with open(f"{i}.pkl", 'rb') as file:
                    acc, reg =pickle.load(file)
                for x in acc:
                    accumulators[x] += acc[x]
                registry.merge_graph(reg)
                os.remove(f"{i}.pkl")
                
    
    # Compute LF(G) from the accumulators
    results = {}
    for cert_G, acc_value in accumulators.items():
        lf_value = acc_value / k
        results[cert_G] = round(lf_value)
    
    # Save to cache if enabled
    if use_caching and cache_file:
        with open(f"{cache_file}_k{max_nodes}_{k}.pkl", 'wb') as f:
            pickle.dump(results, f)
        registry.save_to_file(f"{cache_file}_registry_k{max_nodes}_{k}.pkl")
    
    print(f"Completed k={k} in {time.time() - start_time:.2f} seconds")
    return results, registry


def save_isomorphism_classes(registry, LF_values, k, output_dir="isomorphism_classes"):
    """
    Save isomorphism classes of k-regular graphs organized by vertex count.
    
    Args:
        registry: GraphRegistry object containing the graphs
        LF_values: Dictionary mapping certificates to LF values
        k: Regularity of the graphs
        output_dir: Directory to save the output files
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize graphs by vertex count
    vertex_count_to_graphs = defaultdict(list)
    
    for cert, lf in LF_values.items():
        G = registry.get_graph(cert)
        if G:
            n_vertices = G.number_of_nodes()
            
            # Convert certificate to hex string for readability
            cert_hex = cert.hex()
            
            # Analyze graph structure
            graph_data = {
                "certificate_hex": cert_hex,
                "lf_value": lf,
                "edge_count": G.number_of_edges(),
                "edges": list(G.edges()),
                "is_bipartite": nx.is_bipartite(G),
                "is_connected": nx.is_connected(G),
                "component_sizes": [len(comp) for comp in nx.connected_components(G)],
                "automorphism_count": registry.get_aut_size(cert)
            }
            
            
            # Add graph to the appropriate list
            vertex_count_to_graphs[n_vertices].append(graph_data)
    
    # Calculate sum statistics for each vertex count
    sum_stats = calculate_sum_statistics(registry, LF_values, max(vertex_count_to_graphs.keys()))
    
    # Save summary file
    summary_data = {
        "regularity": k,
        "total_isomorphism_classes": len(LF_values),
        "vertex_counts": {
            str(n): len(graphs) for n, graphs in vertex_count_to_graphs.items()
        },
        "sum_statistics": {
            str(n): sum_value for n, sum_value in sum_stats.items()
        }
    }
    
    with open(f"{output_dir}/summary_k{k}.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed files for each vertex count
    for n_vertices, graphs in vertex_count_to_graphs.items():
        # Sort graphs by LF value
        graphs.sort(key=lambda g: g["lf_value"], reverse=True)
        
        # Create output file
        filename = f"{output_dir}/k{k}_n{n_vertices}.json"
        # with open(filename, 'w') as f:
        #     json.dump(graphs, f, indent=2)
        
        print(f"Saved {len(graphs)} isomorphism classes of {k}-regular graphs on {n_vertices} vertices to {filename}")
    
    return summary_data

def calculate_sum_statistics(registry, LF_values, k):
    """
    Calculate the sum Σ (n!/(|Aut(G)|) * LF(G)) for each vertex count,
    where G ranges over all non-isomorphic graphs with that vertex count.
    
    Args:
        registry: GraphRegistry object containing the graphs
        LF_values: Dictionary mapping certificates to LF values
        max_nodes: Maximum number of nodes considered
        
    Returns:
        Dictionary mapping vertex counts to the calculated sums
    """
    # Group graphs by vertex count
    vertex_count_to_graphs = defaultdict(list)
    
    for cert, lf in LF_values.items():
        G = registry.get_graph(cert)
        if G:
            n_vertices = G.number_of_nodes()
            aut_size = registry.get_aut_size(cert)
            vertex_count_to_graphs[n_vertices].append((cert, lf, aut_size))
    
    # print(vertex_count_to_graphs)
    # Calculate sum for each vertex count
    results = {}
    
    for n_vertices, graph_data in vertex_count_to_graphs.items():
        n_factorial = factorial(n_vertices)
        sum_value = 0
        
        for cert, lf, aut_size in graph_data:
            term = round(n_factorial * lf) // round(aut_size)
            sum_value += term
            # print(f"  Graph {cert.hex()[:8]}: n!/{aut_size} * {lf} = {term}")
        
        results[n_vertices] = sum_value
    # print(results)
    return results

def save_sum_statistics(sums_by_k_n, output_dir="statistics"):
    """
    Save the calculated sum statistics to files.
    
    Args:
        sums_by_k_n: Dictionary mapping (k, n) pairs to sums
        output_dir: Directory to save the output files
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize data by k (regularity)
    data_by_k = defaultdict(dict)
    
    for (k, n), sum_value in sums_by_k_n.items():
        data_by_k[k][n] = sum_value
    
    # Save summary file with all data
    all_data = {
        "description": "Sum of (n!/(|Aut(G)|) * LF(G)) for each regularity k and vertex count n",
        "data": {
            str(k): {
                str(n): sum_value for n, sum_value in values.items()
            } for k, values in data_by_k.items()
        }
    }
    
    with open(f"{output_dir}/all_sums.json", 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Save separate files for each k
    for k, values in data_by_k.items():
        k_data = {
            "regularity": k,
            "sums_by_vertex_count": {
                str(n): sum_value for n, sum_value in sorted(values.items())
            }
        }
        
        with open(f"{output_dir}/sums_k{k}.json", 'w') as f:
            json.dump(k_data, f, indent=2)
    
    print(f"Saved sum statistics to {output_dir}/")
    
    return all_data

def main():
    """
    Example: Compute LF values for k-regular one-factorizable graphs
    with up to max_nodes vertices
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute LF values for k-regular one-factorizable graphs')
    parser.add_argument('-k', type=int, default=2, help='Regularity of the graphs')
    parser.add_argument('--max_nodes', type=int, default=12, help='Maximum number of nodes to consider')
    parser.add_argument('--no_cache', action='store_true', help='Disable caching')
    parser.add_argument('--cache_file', type=str, default='lf_cache', help='Cache file prefix')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--output_dir', type=str, default='isomorphism_classes', help='Directory to save isomorphism classes')
    parser.add_argument('--stats_dir', type=str, default='statistics', help='Directory to save statistics')
    parser.add_argument('--validate', action='store_true', help='Validate results against known counts')
    
    args = parser.parse_args()
    
    # Store sums for all (k,n) pairs
    all_sums = {}
    
    # Determine which k values to process
    k_values = [args.k]
    
    for k in k_values:
        print(f"\n{'='*80}\nComputing for k={k}\n{'='*80}")
        print(f"Computing LF values for {k}-regular one-factorizable graphs with at most {args.max_nodes} vertices...")
        
        LF_values, registry = forward_accumulation(
            k, 
            max_nodes=args.max_nodes,
            use_caching=not args.no_cache,
            cache_file=args.cache_file,
            verbose=args.verbose
        )
        
        print(f"\nFound {len(LF_values)} isomorphism classes of {k}-regular one-factorizable graphs")
        
        # Calculate sum statistics
        sum_stats = calculate_sum_statistics(registry, LF_values, k)
        
        # Store in all_sums
        for n, sum_value in sum_stats.items():
            all_sums[(k, n)] = sum_value
        
        # Print sum statistics
        print("\nCalculated Sums by Vertex Count:")
        for n, sum_value in sorted(sum_stats.items()):
            print(f"  n={n}: Σ (n!/(|Aut(G)|) * LF(G)) = {sum_value}")        
        # Count isomorphism classes by vertex count
        vertex_counts = defaultdict(int)
        for cert in LF_values:
            G = registry.get_graph(cert)
            if G:
                vertex_counts[G.number_of_nodes()] += 1
        
        print("\nSummary by number of vertices:")
        for n_vertices, count in sorted(vertex_counts.items()):
            print(f"{n_vertices} vertices: {count} isomorphism classes")
        
        
        # Save isomorphism classes to files
        save_isomorphism_classes(registry, LF_values, k, args.output_dir)
    
    # Save all sum statistics
    save_sum_statistics(all_sums, args.stats_dir)
    
    print("\nSummary of all calculated sums:")
    for k in sorted(set(k for k, _ in all_sums.keys())):
        print(f"\nk={k} (Regularity):")
        for n in sorted(n for kk, n in all_sums.keys() if kk == k):
            print(f"  n={n}: Σ (n!/(|Aut(G)|) * LF(G)) = {all_sums[(k, n)]}")

if __name__ == "__main__":
    main()
