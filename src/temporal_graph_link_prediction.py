import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# GraKeL and additional imports for advanced features
try:
    from grakel import GraphKernel
    from grakel.kernels import VertexHistogram, EdgeHistogram, GraphletSampling, RandomWalk, ShortestPath
    from grakel.utils import graph_from_networkx
    GRAKEL_AVAILABLE = True
except ImportError:
    print("Warning: GraKeL not available. Install with: pip install grakel")
    GRAKEL_AVAILABLE = False

from itertools import combinations
from scipy import sparse
from scipy.stats import entropy


class TemporalLinkPredictor:
    def __init__(self, decay=0.3, strategy="weighted", seed=17, lookback=5):
        self.decay = decay
        self.strategy = strategy
        self.seed = seed
        self.lookback = lookback
        
        print(f"Initialized: decay={decay}, strategy={strategy}, lookback={lookback}")
    
    def extract_features(self, samples):
        print(f"Processing {len(samples)} samples...")
        
        samples_by_time = self._group_by_time(samples)
        

        features = {
            'neighborhood': self._neighborhood_features(samples),
            'distance': self._distance_features(samples),
            'structural': self._structural_features(samples),
            'temporal_trends': self._temporal_trends_features(samples, samples_by_time),
            'temporal_local': self._temporal_local_features(samples, samples_by_time),
            'wl_patterns': self._wl_features(samples),
            'temporal_subgraph': self._temporal_subgraph_features(samples, samples_by_time),
            'temporal_embedding': self._temporal_embedding_features(samples, samples_by_time),
        }

        return features
    
    def _group_by_time(self, samples):
        from collections import defaultdict
        grouped = defaultdict(list)
        for sample in samples:
            grouped[sample['snapshot_id']].append(sample)
        return grouped
    
    def _temporal_trends_features(self, samples, samples_by_time):    
        features = []
        for sample in samples:
            features.append(self._compute_node_pair_trends(sample, samples_by_time))
        
        return np.array(features)
    
    def _temporal_local_features(self, samples, samples_by_time):   
        features = []
        for sample in samples:
            features.append(self._compute_node_pair_trends_v2(sample, samples_by_time))
        
        return np.array(features)
    
    def _compute_node_pair_trends(self, sample, samples_by_time):
        current_time = sample['snapshot_id']
        u, v = sample['node_u'], sample['node_v']
        
        cn_hist = []  # common neighbors over time
        dist_hist = []  # distance over time
        deg_hist = []  # degree sum over time
        
        # Look back through recent snapshots
        time_steps = sorted(samples_by_time.keys())
        current_idx = time_steps.index(current_time) if current_time in time_steps else len(time_steps)
        
        for i in range(max(0, current_idx - self.lookback), current_idx):
            past_time = time_steps[i]
            if past_time in samples_by_time and samples_by_time[past_time]:
                past_graph = samples_by_time[past_time][0]['graph']
                
                if u in past_graph and v in past_graph:
                    u_neighbors = set(past_graph.neighbors(u))
                    v_neighbors = set(past_graph.neighbors(v))
                    cn_hist.append(len(u_neighbors & v_neighbors))
                    
                    try:
                        dist_hist.append(nx.shortest_path_length(past_graph, u, v))
                    except:
                        dist_hist.append(999)  # no path
                    
                    deg_hist.append(past_graph.degree(u) + past_graph.degree(v))
        
        # Compute trends (slope) and averages
        if len(cn_hist) >= 2:
            cn_trend = (cn_hist[-1] - cn_hist[0]) / len(cn_hist)
            cn_mean = np.mean(cn_hist)
        else:
            cn_trend = cn_mean = 0.0
        
        if len(dist_hist) >= 2:
            dist_trend = -(dist_hist[-1] - dist_hist[0]) / len(dist_hist)  # negative = getting closer
            dist_mean = np.mean(dist_hist)
        else:
            dist_trend = 0.0
            dist_mean = 999.0
        
        if len(deg_hist) >= 2:
            deg_trend = (deg_hist[-1] - deg_hist[0]) / len(deg_hist)
            deg_mean = np.mean(deg_hist)
        else:
            deg_trend = deg_mean = 0.0
        
        return [
            cn_trend, cn_mean, dist_trend, 1.0 / (dist_mean + 1),
            deg_trend, deg_mean / 100, len(cn_hist)
        ]
    
    def _compute_node_pair_trends_v2(self, sample, samples_by_time):
        current_time = sample['snapshot_id']
        u, v = sample['node_u'], sample['node_v']
        
        clust_hist = []  # clustering coefficient over time
        dens_hist = []   # local density over time
        tri_hist = []    # triangles over time
        
        time_steps = sorted(samples_by_time.keys())
        current_idx = time_steps.index(current_time) if current_time in time_steps else len(time_steps)
        
        for i in range(max(0, current_idx - self.lookback), current_idx):
            past_time = time_steps[i]
            if past_time in samples_by_time and samples_by_time[past_time]:
                past_graph = samples_by_time[past_time][0]['graph']
                
                if u in past_graph and v in past_graph:
                    # Average clustering of u and v
                    try:
                        u_clust = nx.clustering(past_graph, u)
                        v_clust = nx.clustering(past_graph, v)
                        clust_hist.append((u_clust + v_clust) / 2)
                    except:
                        clust_hist.append(0.0)
                    
                    # Density of neighborhood around u,v
                    neighbors = set(past_graph.neighbors(u)) | set(past_graph.neighbors(v))
                    neighbors.add(u)
                    neighbors.add(v)
                    if len(neighbors) > 1:
                        subgraph = past_graph.subgraph(neighbors)
                        n_nodes = len(subgraph.nodes())
                        n_edges = len(subgraph.edges())
                        max_edges = n_nodes * (n_nodes - 1) / 2
                        dens_hist.append(n_edges / max_edges if max_edges > 0 else 0)
                    else:
                        dens_hist.append(0.0)
                    
                    # Triangles involving u or v
                    triangles = sum(1 for n in past_graph.neighbors(u) if n in past_graph.neighbors(v))
                    tri_hist.append(triangles)
        
        # Compute trends and averages
        if len(clust_hist) >= 2:
            clust_trend = (clust_hist[-1] - clust_hist[0]) / len(clust_hist)
            clust_mean = np.mean(clust_hist)
        else:
            clust_trend = clust_mean = 0.0
        
        if len(dens_hist) >= 2:
            dens_trend = (dens_hist[-1] - dens_hist[0]) / len(dens_hist)
            dens_mean = np.mean(dens_hist)
        else:
            dens_trend = dens_mean = 0.0
        
        if len(tri_hist) >= 2:
            tri_trend = (tri_hist[-1] - tri_hist[0]) / len(tri_hist)
            tri_mean = np.mean(tri_hist)
        else:
            tri_trend = tri_mean = 0.0
        
        return [
            clust_trend, clust_mean, dens_trend, dens_mean,
            tri_trend, tri_mean, len(clust_hist)
        ]
    
    
    def _wl_features(self, samples):  
        features = []
        for sample in samples:
            graph = sample['graph']
            u, v = sample['node_u'], sample['node_v']
            
            if len(graph.nodes()) == 0:
                features.append([0] * 15)
                continue
            
            subgraph = self._get_subgraph(graph, u, v, radius=2)
            
            if len(subgraph.nodes()) == 0:
                features.append([0] * 15)
                continue
            
            wl_feats = self._do_wl(subgraph, u, v, iterations=3)
            wl_feats.append(np.exp(-self.decay * sample['snapshot_id']))
            features.append(wl_feats)
        
        return np.array(features)
    
    def _get_subgraph(self, graph, u, v, radius=2):
        if u not in graph or v not in graph:
            return nx.Graph()
        
        nodes = set()
        
        if u in graph:
            try:
                u_nbrs = nx.single_source_shortest_path_length(graph, u, cutoff=radius)
                nodes.update(u_nbrs.keys())
            except:
                nodes.add(u)
        
        if v in graph:
            try:
                v_nbrs = nx.single_source_shortest_path_length(graph, v, cutoff=radius)
                nodes.update(v_nbrs.keys())
            except:
                nodes.add(v)
        
        return graph.subgraph(nodes).copy() if nodes else nx.Graph()
    
    def _do_wl(self, graph, u, v, iterations=3):
        #Weisfeiler-Lehman algorithm to capture structural patterns
        if len(graph.nodes()) == 0:
            return [0] * 14
        
        # Start with node degrees as initial labels
        labels = {node: graph.degree(node) for node in graph.nodes()}
        all_hists = []
        
        for i in range(iterations):
            # Create histogram of current label distribution
            label_counts = Counter(labels.values())
            hist = [label_counts.get(i, 0) for i in range(max(label_counts.keys()) + 1)] if label_counts else [0]
            
            if len(hist) < 4:
                hist.extend([0] * (4 - len(hist)))
            else:
                hist = hist[:4]
            
            all_hists.extend(hist)
            
            # Refine labels: combine node's label with neighbor labels
            new_labels = {}
            for node in graph.nodes():
                nbr_labels = sorted([labels[nbr] for nbr in graph.neighbors(node)])
                label_str = f"{labels[node]}_{','.join(map(str, nbr_labels))}"
                new_labels[node] = hash(label_str) % 10000
            
            labels = new_labels
        
        # Pad or truncate to fixed size
        if len(all_hists) < 12:
            all_hists.extend([0] * (12 - len(all_hists)))
        else:
            all_hists = all_hists[:12]
        
        # Add final labels for u and v
        u_label = labels.get(u, 0) if u in graph else 0
        v_label = labels.get(v, 0) if v in graph else 0
        
        return all_hists + [u_label, v_label]
    
    def _neighborhood_features(self, samples):
        features = []
        for sample in samples:
            graph = sample['graph']
            u, v = sample['node_u'], sample['node_v']
            
            if u in graph and v in graph:
                u_nbrs = set(graph.neighbors(u))
                v_nbrs = set(graph.neighbors(v))
                
                common = len(u_nbrs & v_nbrs)
                total = len(u_nbrs | v_nbrs)
                jaccard = common / max(1, total)
                
                aa = sum(1.0 / np.log(graph.degree(nbr)) for nbr in (u_nbrs & v_nbrs) if graph.degree(nbr) > 1)
                u_deg, v_deg = graph.degree(u), graph.degree(v)
                
                try:
                    u_clust = nx.clustering(graph, u)
                    v_clust = nx.clustering(graph, v)
                except:
                    u_clust = v_clust = 0
                
                u_nbr_degs = [graph.degree(n) for n in u_nbrs]
                v_nbr_degs = [graph.degree(n) for n in v_nbrs]
                
                u_avg_conn = np.mean(u_nbr_degs) if u_nbr_degs else 0
                v_avg_conn = np.mean(v_nbr_degs) if v_nbr_degs else 0
                
                ra = sum(1.0 / max(1, graph.degree(n)) for n in (u_nbrs & v_nbrs))
                temp_weight = np.exp(-self.decay * sample['snapshot_id'])
                
                feats = [common, jaccard, aa, u_deg, v_deg, u_deg*v_deg, 
                        u_clust, v_clust, u_avg_conn, v_avg_conn, ra, temp_weight]
            else:
                feats = [0] * 12
            
            features.append(feats)
        
        return np.array(features)
    
    def _distance_features(self, samples):
        features = []
        for sample in samples:
            graph = sample['graph']
            u, v = sample['node_u'], sample['node_v']
            
            try:
                if u in graph and v in graph and nx.has_path(graph, u, v):
                    sp = nx.shortest_path_length(graph, u, v)
                else:
                    sp = 999
            except:
                sp = 999
            
            try:
                u_avg = np.mean(list(nx.shortest_path_length(graph, u).values())) if u in graph else 0
            except:
                u_avg = 0
            
            try:
                v_avg = np.mean(list(nx.shortest_path_length(graph, v).values())) if v in graph else 0
            except:
                v_avg = 0
            
            try:
                if graph.number_of_nodes() > 1 and nx.is_connected(graph):
                    g_avg = nx.average_shortest_path_length(graph)
                else:
                    g_avg = 0
            except:
                g_avg = 0
            
            temp_weight = np.exp(-self.decay * sample['snapshot_id'])
            feats = [sp, u_avg, v_avg, temp_weight, g_avg]
            features.append(feats)
        
        return np.array(features)
    
    def _structural_features(self, samples):
        features = []
        for sample in samples:
            graph = sample['graph']
            u, v = sample['node_u'], sample['node_v']
            
            u_deg = graph.degree(u) if u in graph else 0
            v_deg = graph.degree(v) if v in graph else 0
            
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            density = n_edges / max(1, n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
            
            try:
                avg_clust = nx.average_clustering(graph) if n_nodes > 0 else 0
            except:
                avg_clust = 0
            
            temp_pos = sample['snapshot_id'] * np.exp(-self.decay * sample['snapshot_id'])
            feats = [u_deg, v_deg, n_nodes, n_edges, density, avg_clust, temp_pos]
            features.append(feats)
        
        return np.array(features)

    def _embedding_features(self, samples):
        #Extract node embedding-based features using simple graph embeddings
        features = []
        
        for sample in samples:
            graph = sample['graph']
            u, v = sample['node_u'], sample['node_v']
            
            if len(graph.nodes()) < 2:
                features.append([0] * 8)
                continue
            
            try:
                embedding_feats = []
                
                # Simple position-based embeddings
                
                # Degree-based embedding
                if u in graph and v in graph:
                    u_deg = graph.degree(u)
                    v_deg = graph.degree(v)
                    deg_similarity = 2 * min(u_deg, v_deg) / (u_deg + v_deg + 1)
                    deg_product = u_deg * v_deg
                    embedding_feats.extend([deg_similarity, deg_product])
                else:
                    embedding_feats.extend([0, 0])
                
                # Clustering-based embedding
                try:
                    u_clust = nx.clustering(graph, u) if u in graph else 0
                    v_clust = nx.clustering(graph, v) if v in graph else 0
                    clust_similarity = 2 * min(u_clust, v_clust) / (u_clust + v_clust + 1e-8)
                    clust_product = u_clust * v_clust
                    embedding_feats.extend([clust_similarity, clust_product])
                except:
                    embedding_feats.extend([0, 0])
                
                # Neighbor-based embedding (Jaccard-like)
                if u in graph and v in graph:
                    u_neighbors = set(graph.neighbors(u))
                    v_neighbors = set(graph.neighbors(v))
                    
                    jaccard = len(u_neighbors & v_neighbors) / len(u_neighbors | v_neighbors) if u_neighbors | v_neighbors else 0
                    cosine_sim = len(u_neighbors & v_neighbors) / (len(u_neighbors) * len(v_neighbors))**0.5 if u_neighbors and v_neighbors else 0
                    
                    embedding_feats.extend([jaccard, cosine_sim])
                else:
                    embedding_feats.extend([0, 0])
                
                # Random walk-based similarity (approximate)
                try:
                    if u in graph and v in graph and nx.has_path(graph, u, v):
                        # Simple approximation: inverse of shortest path
                        sp_length = nx.shortest_path_length(graph, u, v)
                        rw_similarity = 1.0 / (sp_length + 1)
                        # Path diversity
                        try:
                            all_paths = list(nx.all_simple_paths(graph, u, v, cutoff=3))
                            path_diversity = min(len(all_paths), 10) / 10.0  # Normalize
                        except:
                            path_diversity = 0
                    else:
                        rw_similarity = 0
                        path_diversity = 0
                    embedding_feats.extend([rw_similarity, path_diversity])
                except:
                    embedding_feats.extend([0, 0])
                
                features.append(embedding_feats)
                
            except Exception as e:
                features.append([0] * 8)
        
        return np.array(features)

    def _similarity_features(self, samples):
        # Extract various node similarity measures
        features = []
        
        for sample in samples:
            graph = sample['graph']
            u, v = sample['node_u'], sample['node_v']
            
            if len(graph.nodes()) < 2:
                features.append([0] * 7)
                continue
            
            try:
                similarity_feats = []
                
                if u in graph and v in graph:
                    u_neighbors = set(graph.neighbors(u))
                    v_neighbors = set(graph.neighbors(v))
                    
                    # Common Neighbors
                    common_neighbors = len(u_neighbors & v_neighbors)
                    similarity_feats.append(common_neighbors)
                    
                    # Adamic-Adar Index
                    aa_score = 0
                    for common in u_neighbors & v_neighbors:
                        if graph.degree(common) > 1:
                            aa_score += 1.0 / np.log(graph.degree(common))
                    similarity_feats.append(aa_score)
                    
                    # Resource Allocation Index
                    ra_score = 0
                    for common in u_neighbors & v_neighbors:
                        if graph.degree(common) > 0:
                            ra_score += 1.0 / graph.degree(common)
                    similarity_feats.append(ra_score)
                    
                    # Preferential Attachment
                    pa_score = len(u_neighbors) * len(v_neighbors)
                    similarity_feats.append(pa_score)
                    
                    # Hub Promoted Index
                    hub_promoted = common_neighbors / min(len(u_neighbors), len(v_neighbors)) if min(len(u_neighbors), len(v_neighbors)) > 0 else 0
                    similarity_feats.append(hub_promoted)
                    
                    # Hub Depressed Index  
                    hub_depressed = common_neighbors / max(len(u_neighbors), len(v_neighbors)) if max(len(u_neighbors), len(v_neighbors)) > 0 else 0
                    similarity_feats.append(hub_depressed)
                    
                    # Leicht-Holme-Newman Index
                    lhn_score = common_neighbors / (len(u_neighbors) * len(v_neighbors)) if len(u_neighbors) * len(v_neighbors) > 0 else 0
                    similarity_feats.append(lhn_score)
                    
                else:
                    similarity_feats = [0] * 7
                
                features.append(similarity_feats)
                
            except Exception as e:
                features.append([0] * 7)
        
        return np.array(features)

    def _temporal_subgraph_features(self, samples, samples_by_time):
        # Extract temporal evolution of subgraphs around node pairs
        features = []
        
        for sample in samples:
            current_time = sample['snapshot_id']
            u, v = sample['node_u'], sample['node_v']
            
            try:
                temporal_feats = []
                
                # Look back through time to track subgraph evolution
                time_steps = sorted(samples_by_time.keys())
                current_idx = time_steps.index(current_time) if current_time in time_steps else len(time_steps)
                
                # Track subgraph properties over time
                subgraph_sizes = []
                subgraph_densities = []
                subgraph_edges = []
                subgraph_triangles = []
                
                for i in range(max(0, current_idx - self.lookback), current_idx):
                    past_time = time_steps[i]
                    if past_time in samples_by_time and samples_by_time[past_time]:
                        past_graph = samples_by_time[past_time][0]['graph']
                        
                        # Extract subgraph at this time step
                        past_subgraph = self._get_subgraph(past_graph, u, v, radius=2)
                        
                        if len(past_subgraph.nodes()) >= 2:
                            # Size evolution
                            subgraph_sizes.append(len(past_subgraph.nodes()))
                            subgraph_edges.append(len(past_subgraph.edges()))
                            
                            # Density evolution
                            n_nodes = len(past_subgraph.nodes())
                            n_edges = len(past_subgraph.edges())
                            max_edges = n_nodes * (n_nodes - 1) / 2
                            density = n_edges / max_edges if max_edges > 0 else 0
                            subgraph_densities.append(density)
                            
                            # Triangle count evolution
                            triangles = 0
                            if u in past_subgraph and v in past_subgraph:
                                u_neighbors = set(past_subgraph.neighbors(u))
                                v_neighbors = set(past_subgraph.neighbors(v))
                                triangles = len(u_neighbors & v_neighbors)
                            subgraph_triangles.append(triangles)
                
                # Compute temporal features from evolution
                if len(subgraph_sizes) >= 2:
                    # Growth trends
                    size_trend = (subgraph_sizes[-1] - subgraph_sizes[0]) / len(subgraph_sizes)
                    edge_trend = (subgraph_edges[-1] - subgraph_edges[0]) / len(subgraph_edges)
                    density_trend = (subgraph_densities[-1] - subgraph_densities[0]) / len(subgraph_densities)
                    triangle_trend = (subgraph_triangles[-1] - subgraph_triangles[0]) / len(subgraph_triangles)
                    
                    # Stability measures
                    size_stability = 1.0 / (1.0 + np.var(subgraph_sizes))
                    density_stability = 1.0 / (1.0 + np.var(subgraph_densities))
                    
                    # Recent vs historical comparison
                    recent_avg_density = np.mean(subgraph_densities[-2:]) if len(subgraph_densities) >= 2 else 0
                    historical_avg_density = np.mean(subgraph_densities[:-2]) if len(subgraph_densities) > 2 else recent_avg_density
                    density_acceleration = recent_avg_density - historical_avg_density
                    
                    temporal_feats = [
                        size_trend, edge_trend, density_trend, triangle_trend,
                        size_stability, density_stability, density_acceleration,
                        len(subgraph_sizes)  # Number of time points observed
                    ]
                else:
                    temporal_feats = [0] * 8
                
                features.append(temporal_feats)
                
            except Exception as e:
                features.append([0] * 8)
        
        return np.array(features)


    def _temporal_embedding_features(self, samples, samples_by_time):
        # Extract temporal evolution of node embedding similarities
        features = []
        
        for sample in samples:
            current_time = sample['snapshot_id']
            u, v = sample['node_u'], sample['node_v']
            
            try:
                embedding_feats = []
                
                time_steps = sorted(samples_by_time.keys())
                current_idx = time_steps.index(current_time) if current_time in time_steps else len(time_steps)
                
                # Track embedding similarities over time
                degree_similarity_history = []
                jaccard_similarity_history = []
                clustering_similarity_history = []
                
                for i in range(max(0, current_idx - self.lookback), current_idx):
                    past_time = time_steps[i]
                    if past_time in samples_by_time and samples_by_time[past_time]:
                        past_graph = samples_by_time[past_time][0]['graph']
                        
                        if u in past_graph and v in past_graph:
                            # Degree similarity evolution
                            u_deg = past_graph.degree(u)
                            v_deg = past_graph.degree(v)
                            deg_sim = 2 * min(u_deg, v_deg) / (u_deg + v_deg + 1)
                            degree_similarity_history.append(deg_sim)
                            
                            # Jaccard similarity evolution
                            u_neighbors = set(past_graph.neighbors(u))
                            v_neighbors = set(past_graph.neighbors(v))
                            jaccard = len(u_neighbors & v_neighbors) / len(u_neighbors | v_neighbors) if u_neighbors | v_neighbors else 0
                            jaccard_similarity_history.append(jaccard)
                            
                            # Clustering similarity evolution
                            try:
                                u_clust = nx.clustering(past_graph, u)
                                v_clust = nx.clustering(past_graph, v)
                                clust_sim = 2 * min(u_clust, v_clust) / (u_clust + v_clust + 1e-8)
                                clustering_similarity_history.append(clust_sim)
                            except:
                                clustering_similarity_history.append(0)
                
                # Compute temporal embedding features
                if len(degree_similarity_history) >= 2:
                    # Similarity evolution trends
                    deg_sim_trend = (degree_similarity_history[-1] - degree_similarity_history[0]) / len(degree_similarity_history)
                    jaccard_trend = (jaccard_similarity_history[-1] - jaccard_similarity_history[0]) / len(jaccard_similarity_history)
                    clust_sim_trend = (clustering_similarity_history[-1] - clustering_similarity_history[0]) / len(clustering_similarity_history)
                    
                    # Similarity convergence (are they becoming more similar?)
                    recent_similarity = np.mean([degree_similarity_history[-1], jaccard_similarity_history[-1], clustering_similarity_history[-1]])
                    historical_similarity = np.mean([degree_similarity_history[0], jaccard_similarity_history[0], clustering_similarity_history[0]])
                    similarity_convergence = recent_similarity - historical_similarity
                    
                    # Similarity stability
                    deg_stability = 1.0 / (1.0 + np.var(degree_similarity_history))
                    jaccard_stability = 1.0 / (1.0 + np.var(jaccard_similarity_history))
                    
                    embedding_feats = [
                        deg_sim_trend, jaccard_trend, clust_sim_trend,
                        similarity_convergence, deg_stability, jaccard_stability
                    ]
                else:
                    embedding_feats = [0] * 6
                
                features.append(embedding_feats)
                
            except Exception as e:
                features.append([0] * 6)
        
        return np.array(features)
    
    def train(self, samples, labels):

        pos_count = sum(labels)
        print(f"Positive: {pos_count} ({pos_count/len(labels):.1%})")
        
        features = self.extract_features(samples)
        y = np.array(labels)
        
        train_idx, test_idx = train_test_split(
            range(len(samples)), test_size=0.2, random_state=self.seed, stratify=y
        )
        

        
        results = {}
        predictions = {}
        
        # Store models and scalers for TGB compatibility
        self.models = {}
        self.scalers = {}
        
        for name, feat in features.items():

            
            X_train, X_test = feat[train_idx], feat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=self.seed, max_depth=10)
            clf.fit(X_train_scaled, y_train)
            
            # Store model and scaler
            self.models[name] = clf
            self.scalers[name] = scaler
            
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            # Calculate MRR (Mean Reciprocal Rank)
            mrr = self._calculate_mrr(y_test, y_prob)
            
            # Calculate MAP (Mean Average Precision) - another ranking metric
            map_score = average_precision_score(y_test, y_prob)
            
            results[name] = {
                "accuracy": acc, 
                "roc_auc": auc, 
                "mrr": mrr,
                "map": map_score
            }
            predictions[name] = y_prob
            

        ensemble_pred, weights = self._create_ensemble(results, predictions, y[test_idx])
        
        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(ensemble_pred, y[test_idx])
        
        # Evaluate with optimal threshold
        ensemble_pred_binary = (ensemble_pred > optimal_threshold).astype(int)
        ensemble_acc = accuracy_score(y[test_idx], ensemble_pred_binary)
        ensemble_auc = roc_auc_score(y[test_idx], ensemble_pred)
        
        # Also calculate F1 score for imbalanced data
        ensemble_f1 = f1_score(y[test_idx], ensemble_pred_binary)
        
        # Calculate ensemble MRR and MAP
        ensemble_mrr = self._calculate_mrr(y[test_idx], ensemble_pred)
        ensemble_map = average_precision_score(y[test_idx], ensemble_pred)
        
        # Store ensemble weights for TGB compatibility
        self.ensemble_weights = weights
        
        final_results = {
            'ensemble_accuracy': ensemble_acc,
            'ensemble_roc_auc': ensemble_auc,
            'ensemble_f1': ensemble_f1,
            'ensemble_mrr': ensemble_mrr,
            'ensemble_map': ensemble_map,
            'optimal_threshold': optimal_threshold,
            'kernel_results': results,
            'ensemble_weights': weights,
            'best_kernel': max(results.keys(), key=lambda k: results[k]["roc_auc"]),
            'test_size': len(test_idx)
        }
        

        for name, metrics in results.items():
            print(f"  {name}: {metrics['accuracy']:.3f} acc, {metrics['roc_auc']:.3f} auc, {metrics['mrr']:.3f} mrr, {metrics['map']:.3f} map")
        print(f"  ensemble: {ensemble_acc:.3f} acc, {ensemble_auc:.3f} auc, {ensemble_f1:.3f} f1, {ensemble_mrr:.3f} mrr, {ensemble_map:.3f} map")
        print(f"  optimal threshold: {optimal_threshold:.3f}")
        
        return final_results
    
    def _calculate_mrr(self, y_true, y_prob):
        """Calculate Mean Reciprocal Rank for link prediction"""
        # For each positive example, find its rank among all examples
        reciprocal_ranks = []
        
        # Group by positive examples
        positive_indices = np.where(y_true == 1)[0]
        
        if len(positive_indices) == 0:
            return 0.0
        
        # For each positive example, calculate its rank
        for pos_idx in positive_indices:
            pos_score = y_prob[pos_idx]
            
            # Count how many examples have higher scores
            rank = 1 + np.sum(y_prob > pos_score)
            
            # Add reciprocal rank
            reciprocal_ranks.append(1.0 / rank)
        
        return np.mean(reciprocal_ranks)
    
    def _find_optimal_threshold(self, y_prob, y_true):
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
        return thresholds[optimal_idx]
    
    def _create_ensemble(self, results, predictions, y_true):
        """Combine predictions with performance-based weighting"""
        temporal_kernels = {'temporal_trends', 'temporal_local'}
        
        weights = {}
        total_weight = 0
        
        # Weight based on performance and feature type
        for name, perf in results.items():
            auc = perf['roc_auc']
            
            if name in temporal_kernels:
                # Boost temporal features slightly
                if auc > 0.75:
                    weight = auc * 1.5
                elif auc > 0.65:
                    weight = auc * 1.2
                else:
                    weight = auc * 0.8
            else:
                # Standard weighting for static features
                if auc < 0.55:
                    weight = 0.1
                elif auc < 0.65:
                    weight = 0.5
                else:
                    weight = auc
            
            weights[name] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0/len(results) for k in results.keys()}
        
        # Weighted combination
        ensemble_pred = np.zeros(len(y_true))
        for name, weight in weights.items():
            ensemble_pred += weight * predictions[name]
        
        return ensemble_pred, weights


def load_data(file_path, n_snapshots=500):
    print(f"Loading: {file_path}")
    
    try:
        data = pd.read_csv(file_path, sep='\t', header=None, names=['source', 'target', 'timestamp'])
        data = data.sort_values('timestamp')
        print(f"Loaded {len(data)} edges")
    except Exception as e:
        raise FileNotFoundError(f"Error loading {file_path}: {e}")
    
    # make snapshots
    time_range = data['timestamp'].max() - data['timestamp'].min()
    window = time_range / n_snapshots
    
    snapshots = []
    for i in range(n_snapshots):
        start = data['timestamp'].min() + i * window
        end = start + window
        
        edges = data[(data['timestamp'] >= start) & (data['timestamp'] < end)]
        
        if len(edges) > 0:
            g = nx.from_pandas_edgelist(edges, source='source', target='target')
        else:
            g = nx.Graph()
        
        g.graph['snapshot_id'] = i
        g.graph['timestamp'] = start
        snapshots.append(g)
    
    print(f"Created {len(snapshots)} snapshots")
    return snapshots


def make_samples(snapshots, max_samples=15000):

    print("Making samples...")
    
    samples = []
    labels = []
    
    for i in range(len(snapshots) - 1):
        curr = snapshots[i]
        next_g = snapshots[i + 1]
        
        curr_nodes = set(curr.nodes())
        if len(curr_nodes) < 2:
            continue
        
        # Positive samples: edges that actually form
        curr_edges = set(curr.edges())
        next_edges = set(next_g.edges())
        new_edges = next_edges - curr_edges
        
        for u, v in new_edges:
            if u in curr_nodes and v in curr_nodes:
                samples.append({
                    'snapshot_id': i,
                    'node_u': u,
                    'node_v': v,
                    'graph': curr
                })
                labels.append(1)
        
        # Negative samples: node pairs that don't connect
        neg_target = min(len(new_edges) * 2, 100)
        neg_count = 0
        
        nodes = list(curr_nodes)
        np.random.seed(42)
        
        attempts = 0
        max_attempts = neg_target * 10
        while neg_count < neg_target and attempts < max_attempts:
            attempts += 1
            u, v = np.random.choice(nodes, 2, replace=False)
            
            # Skip if already connected or will connect
            if (u, v) not in curr_edges and (v, u) not in curr_edges:
                if (u, v) not in next_edges and (v, u) not in next_edges:
                    samples.append({
                        'snapshot_id': i,
                        'node_u': u,
                        'node_v': v,
                        'graph': curr
                    })
                    labels.append(0)
                    neg_count += 1
        
        if len(samples) >= max_samples:
            break
    
    # Limit total samples
    if len(samples) > max_samples:
        idx = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]
        labels = [labels[i] for i in idx]
    
    pos_count = sum(labels)
    print(f"Created {len(samples)} samples ({pos_count} pos, {pos_count/len(labels):.1%})")
    
    return samples, labels


def load_collegemsg_data():
    data_file = "data/CollegeMsg_edges.txt"
    
    if not Path(data_file).exists():
        print(f"File not found: {data_file}")
        return [], []
    
    snapshots = load_data(data_file, n_snapshots=500)
    samples, labels = make_samples(snapshots, max_samples=15000)
    
    return samples, labels


def run_exp():
    samples, labels = load_collegemsg_data()
    
    if not samples:
        return
    
    # train
    model = TemporalLinkPredictor(decay=0.3, strategy="weighted", seed=42)
    
    start = time.time()
    results = model.train(samples, labels)
    train_time = time.time() - start
    
    # show results
    print("Results")
    rankings = sorted(results['kernel_results'].items(), key=lambda x: x[1]['mrr'], reverse=True)
    
    print("\nRankings (by MRR):")
    for rank, (name, metrics) in enumerate(rankings, 1):
        if name.startswith("temporal"):
            extractor_type = "Temporal"
        elif name == "wl_patterns":
            extractor_type = "WL Patterns"
        else:
            extractor_type = "Static"
        print(f"{rank}. {name} ({extractor_type}): {metrics['accuracy']:.3f} acc, {metrics['roc_auc']:.3f} auc, {metrics['mrr']:.3f} mrr, {metrics['map']:.3f} map")
    
    print(f"\nEnsemble: {results['ensemble_accuracy']:.3f} acc, {results['ensemble_roc_auc']:.3f} auc, {results['ensemble_mrr']:.3f} mrr, {results['ensemble_map']:.3f} map")
    print(f"Time: {train_time:.1f}s")
    print(f"Test samples: {results['test_size']}")
    
    # improvement
    best_acc = max(metrics['accuracy'] for metrics in results['kernel_results'].values())
    improvement = results['ensemble_accuracy'] - best_acc
    
    if improvement > 0:
        print(f"Ensemble improvement: +{improvement:.3f}")
    else:
        print(f"Ensemble vs best: {improvement:.3f}")
    



if __name__ == "__main__":
    import time
    
    print("Temporal Graph Link Prediction")
    print("Loading CollegeMsg dataset...")
    
    run_exp() 