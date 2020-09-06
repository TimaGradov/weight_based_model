import warnings
warnings.filterwarnings('ignore')
from collections import Counter, OrderedDict
import community
import json
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode, plot
init_notebook_mode(connected=True)

from itertools import combinations
from numpy.linalg import norm
from scipy.spatial.distance import cosine, minkowski, jaccard, hamming
from tqdm import tqdm_notebook, tqdm
import random
import json
from functools import partial
from node2vec import Node2Vec
from sklearn.cluster import KMeans

def process_vectors(path):
    with open(path, 'r') as f:
        vect = json.load(f)
    
    clear_vect = {}
    for k, v in vect.items():
        if sum(v) == 0:
           continue
        clear_vect[k] = (np.array(v) > 0).astype(int)
    
    del vect
    
    return clear_vect

def significance_normalization(vect, gamma = 0):
    
    interests_array = np.array(list(vect.values()))
    w_k = 1 - gamma * np.abs(interests_array).sum(0) / interests_array.shape[0]
    
    normalized_vect = {}
    for key, vector in vect.items():
        normalized_vect[key] = vector * w_k
        
    return normalized_vect, w_k

def calc_common_interests_stats(G, vect):
    common_interests_number_list = []
    for a, b in G.edges():
        if (a not in vect) or (b not in vect):
            continue
        edge_score = np.sum(vect[a] * vect[b])
        common_interests_number_list.append(edge_score)
    
    common_interests_number_list = np.array(common_interests_number_list)
    mu = np.mean(common_interests_number_list)
    sigma = np.std(common_interests_number_list)
    return common_interests_number_list, mu, sigma

# Exponential weight
def calc_gaussian_weight(a, b, mu, sigma):
    common_interests = np.sum(np.array(a) * np.array(b))
    answ = np.exp( - (common_interests - mu)**2 / (2*sigma**2))
    
    return answ

def inner_sim(a, b, l1_norm, l2_norm, l2_norm_squared):
    res = np.inner(a, b) /  l2_norm_squared / len(a)
    return res

def cosine_sim(a, b, l1_norm, l2_norm, l2_norm_squared):
    return 1 - cosine(a, b)

def manhattan_sim(a, b, l1_norm, l2_norm, l2_norm_squared):
    return 1 - np.sum(np.abs(a-b)) / l1_norm / len(a)

def euclidean_sim(a, b, l1_norm, l2_norm, l2_norm_squared):
    res = 1 - np.sqrt(np.sum((a-b)**2)) / l2_norm / len(a)**0.5
    return res

def jaccard_sim(a, b, l1_norm, l2_norm, l2_norm_squared):
    return 1 - jaccard(a, b)

# Matching coefficient
def hamming_sim(a, b, l1_norm, l2_norm, l2_norm_squared):
    return 1 - hamming(a, b)


def calculate_similarity(G_n_f, l1_norm, l2_norm, l2_norm_squared, nodes, vect, similarity_metric, first_node):
    similarity_vector = {}
    for second_node in G_n_f.nodes():
        attr_a, attr_b = vect[first_node], vect[second_node]
        topic_similarity = similarity_metric(attr_a, attr_b, l1_norm, l2_norm, l2_norm_squared)
        similarity_vector[second_node] = topic_similarity
    sorted_sim_vect = sorted(similarity_vector.items(), key=lambda kv: kv[1], reverse=True)
    return (first_node, sorted_sim_vect)

def not_fixed_topology_graph(G, vect, omega, similarity_metric, file_name):

    l1_norm = norm(omega, 1)
    l2_norm = norm(omega, 2)
    l2_norm_squared = l2_norm ** 2
    hs = open(file_name,"a")
    similarity_matrix = []
    for node in tqdm_notebook(G.nodes(), total=G.number_of_nodes(), leave=False):
        hs.write(str(calculate_similarity(G, l1_norm, l2_norm, l2_norm_squared, G.nodes(), vect, similarity_metric, node)) + "\n")

    hs.close() 
    return similarity_matrix

def create_weighted_graph_from_two_graphs(G_first, G_second, alpha=0.5):
    G = nx.Graph()
    first_edges = list(G_first.edges)
    first_weight = G_first.size(weight='weight')
    second_edges = list(G_second.edges)
    second_weight = G_second.size(weight='weight')
    for edge in tqdm_notebook(G_second.edges(data=True), total=G_second.number_of_edges(), leave=False):
        G.add_edge(edge[0], edge[1], weight=(edge[2]['weight']/second_weight) * (1-alpha))
    for i in tqdm_notebook(range(G_first.number_of_edges()), total=G_first.number_of_edges(), leave=False):
        if G.has_edge(first_edges[i][0], first_edges[i][1]):
            current_weight = G.get_edge_data(first_edges[i][0], first_edges[i][1])['weight']
            G[first_edges[i][0]][first_edges[i][1]]['weight'] = current_weight + (1/first_weight) * alpha
        else:
            G.add_edge(first_edges[i][0], first_edges[i][1], weight=(1/first_weight) * alpha)
    return G

# Gaussian_weighting
def create_weighted_graph(G, vect, mu, sigma, omega, similarity_metric,
                          alpha=0.5, gaussian_weighting=False):
    
    G_w = G.copy()
    representative_edges_num = 0
    similarity_metric_sum = 0
    mixed_weights_arr = []
    
    l1_norm = norm(omega, 1)
    l2_norm = norm(omega, 2)
    l2_norm_squared = l2_norm ** 2

    for a, b in tqdm_notebook(G_w.edges(), total=G_w.number_of_edges(), leave=False):
        if (a not in vect) or (b not in vect):
            
            continue
        representative_edges_num += 1
        
        attr_a, attr_b = vect[a], vect[b]
        topic_similarity = similarity_metric(attr_a, attr_b, l1_norm, l2_norm, l2_norm_squared)
        if gaussian_weighting == True:
            
            w_g = calc_gaussian_weight(attr_a, attr_b, mu, sigma)
            topic_gaussian_similarity = topic_similarity * w_g
            similarity_metric_sum += topic_gaussian_similarity
        
        else:
            
            similarity_metric_sum += topic_similarity
    
    for a, b in tqdm_notebook(G_w.edges(), total=G_w.number_of_edges(), leave=False):
        
        if (a not in vect) or (b not in vect):
            G_w[a][b]['weight'] = alpha 
            G_w[b][a]['weight'] = alpha
            continue
        
        attr_a, attr_b = vect[a], vect[b]
        topic_similarity = similarity_metric(attr_a, attr_b, l1_norm, l2_norm, l2_norm_squared)
        
        if gaussian_weighting == True:
            
            w_g = calc_gaussian_weight(attr_a, attr_b, mu, sigma)
            topic_gaussian_similarity = topic_similarity * w_g
        
            mixed_weight = (alpha * G_w[a][b]['weight'] / representative_edges_num + (1-alpha) * topic_gaussian_similarity / similarity_metric_sum) * representative_edges_num
            
        else:
            mixed_weight = (alpha * G_w[a][b]['weight'] / representative_edges_num + (1-alpha) * topic_similarity / similarity_metric_sum) * representative_edges_num
        
        mixed_weight = max(mixed_weight, 0) 
        G_w[a][b]['weight'] = mixed_weight
        G_w[b][a]['weight'] = mixed_weight
        
        mixed_weights_arr.append(mixed_weight)
    

    mixed_weights_stats = pd.Series(mixed_weights_arr).describe().values[1:]
    
    
    return G_w, mixed_weights_stats

def entropy(G, clusters, vect):

    for v in vect.values():
        attr_num = v.shape[0]
        break
    
    entropy = 0
    active_nodes = 0
    
    for cluster in clusters:
        cluster_matr = []
        for node in cluster:
            if node not in vect:
                continue
            cluster_matr.append(vect[node])
        cluster_matr = np.array(cluster_matr)
        
        ones_count = cluster_matr.sum(0)
        zeros_count = cluster_matr.shape[0] - ones_count
        ones_proportion = ones_count / cluster_matr.shape[0] + 1e-5
        zeros_proportion = zeros_count / cluster_matr.shape[0] + 1e-5
        
        entropy_per_cluster = -np.sum(ones_proportion * np.log2(ones_proportion) + \
                            zeros_proportion * np.log2(zeros_proportion)) / attr_num 
        
        entropy += entropy_per_cluster
        active_nodes += cluster_matr.shape[0]
    
    entropy_avg = entropy / active_nodes
    
    return entropy_avg

def calc_intra_cluster_density(clusters, G):
    intra_cluster_density = 0
    cluster_density = 0
    for c in clusters:
        nodes_clust = set(c)
        e_in = 0
        for node in c:
            nbr = set(G.neighbors(node))
            inside_cluster = len(nbr.intersection(nodes_clust))
            e_in += inside_cluster / 2
        if (len(nodes_clust) == 1):
            cluster_density = 1
        else:
            cluster_density = e_in / (len(nodes_clust) * (len(nodes_clust) - 1) / 2)
        intra_cluster_density += cluster_density
    return intra_cluster_density / len(clusters)

def calc_inter_cluster_density(clusters, G):
    inter_cluster_density = 0
    cluster_density = 0
    for c in clusters:
        nodes_clust = set(c)
        e_out = 0
        for node in c:
            nbr = set(G.neighbors(node))
            inside_cluster = len(nbr.intersection(nodes_clust))
            outside_cluster = (len(nbr) - inside_cluster)
            
            e_out += outside_cluster / 2
        if (len(nodes_clust) * (G.number_of_nodes() - len(nodes_clust)) == 0):
            cluster_density = e_out
        else:
            cluster_density = e_out /  (len(nodes_clust) * (G.number_of_nodes() - len(nodes_clust)))
        inter_cluster_density += cluster_density
    return inter_cluster_density / len(clusters)

def calc_cluster_density(clusters, G):
    cluster_density = 0
    for c in clusters:
        nodes_clust = set(c)
        e_in = 0
        e_out = 0
        for node in c:
            nbr = set(G.neighbors(node))
            inside_cluster = len(nbr.intersection(nodes_clust))
            outside_cluster = (len(nbr) - inside_cluster)
            
            e_in += inside_cluster / 2
            e_out += outside_cluster / 2
    cluster_density = (e_in - e_out) /  G.number_of_edges()
    return cluster_density

def calc_cluster_harmony(clusters, G, vect, similarity_metric, omega):
    l1_norm = norm(omega, 1)
    l2_norm = norm(omega, 2)
    l2_norm_squared = l2_norm ** 2

    cluster_harmony = 0
    h_in = 0
    h_out = 0

    for c in clusters:
        nodes_clust = set(c)
        for node in c:
            nbr = set(G.neighbors(node))
            inside_nbr = nbr.intersection(nodes_clust)
            outside_nbr = nbr.difference(nodes_clust)
            for in_node in inside_nbr:
                attr_a, attr_b = vect[node], vect[in_node]
                node_sim = similarity_metric(attr_a, attr_b, l1_norm, l2_norm, l2_norm_squared)
                h_in += node_sim / 2
            for out_node in outside_nbr:
                attr_a, attr_b = vect[node], vect[out_node]
                node_sim = similarity_metric(attr_a, attr_b, l1_norm, l2_norm, l2_norm_squared)
                h_out += node_sim / 2
    cluster_harmony = (h_in - h_out) / G.number_of_edges()
    return cluster_harmony

def calc_modularity_density(clusters, G):
    modularity_density = 0
    for c in clusters:
        nodes_clust = set(c)
        e_in = 0
        e_out = 0
        for node in c:
            nbr = set(G.neighbors(node))
            inside_cluster = len(nbr.intersection(nodes_clust))
            outside_cluster = (len(nbr) - inside_cluster)
            
            e_in += inside_cluster / 2
            e_out += outside_cluster / 2
        
        nodes_num = len(nodes_clust)
        
        d_in = 2 * e_in / nodes_num
        d_out = 2 * e_out / nodes_num
        d_per_cluster = (d_in - d_out) 
        
        modularity_density += d_per_cluster
    modularity_density /= len(clusters)
    return modularity_density

def permanence(partition, G):
    perm = 0
    olo = 0
    for node, comm in partition.items():
        
        degree = G.degree(node)
        other_communities_nbr = []
        same_community_nbr = []
        for nbr in G.neighbors(node):
            nbr_comm = partition[nbr]
            if nbr_comm == comm:
                same_community_nbr.append(nbr)
            else:
                other_communities_nbr.append(nbr_comm)
         
        i_v = len(same_community_nbr)
        if len(other_communities_nbr) > 1:
            e_max = Counter(other_communities_nbr).most_common(1)[0][1]
        else:
            e_max = 1
        
        c_in = local_clustering_coefficinet(same_community_nbr, G)
        
        perm_node = i_v / e_max / degree - (1 - c_in)
        perm += perm_node
    perm /= G.number_of_nodes()
    return perm
             
def local_clustering_coefficinet(nbrs, G):
    numerator = 0
    denominator = 0
    for edge in combinations(nbrs, 2):
        numerator += int(edge in G.edges())
        denominator += 1
    denominator = max(denominator, 1)
    return numerator / denominator

def node2vec_partition(G_w):
    node2vec = Node2Vec(G_w, dimensions=128, num_walks=100, 
                    workers=1)
    model = node2vec.fit(window=6, workers=1)
    kmeans = KMeans(n_clusters=10, n_jobs=1)
    nodes = list(G_w.nodes())
    node_embeddings = np.array([model.wv.get_vector(node) 
                                for node in nodes])

    kmeans.fit(node_embeddings)
    partition = dict(zip(nodes, kmeans.labels_.tolist()))
    return partition

def generate_report(clusters, vect, names):
    number_of_active_nodes = len(vect)
    attribute_full_matrix = np.array(list(vect.values()))
    attribute_full_freqs = attribute_full_matrix.sum(0) / attribute_full_matrix.shape[0]
    report = []
    
    for n, cluster in enumerate(clusters):
        
        attribute_cluster_matrix = []
        for node in cluster:
            if node not in vect:
                continue
            attribute_cluster_matrix.append(vect[node])
        
        if len(attribute_cluster_matrix) < 15:
            continue
        
        attribute_cluster_matrix = np.array(attribute_cluster_matrix)
        attribute_cluster_freqs = attribute_cluster_matrix.sum(0) / attribute_cluster_matrix.shape[0]
        
        cluster_coefs = attribute_cluster_freqs / attribute_full_freqs
        cluster_coefs = np.round(cluster_coefs, 2)
        cluster_coefs = [len(attribute_cluster_matrix)] + cluster_coefs.tolist()
        report.append(cluster_coefs)
        
    cols = ['nodes_number'] + list(names)
    report_df = pd.DataFrame(report, columns=cols)
    return report_df

def calc_tau(report_df):
    zz = report_df.iloc[:, 1:]
    k = zz / zz.mean()
    cumul_tau = 0
    log = []
    for n, row in k.iterrows():
        a = sum([i if i > 1 else 0 for i in row ])
        b = np.nansum(row.values)
        cumul_tau += a / b
        log.append(a / b)
    if (k.shape[0] != 0):
        cumul_tau /= k.shape[0]
    return cumul_tau, dict(enumerate(log))

def vizualize_report(report_df, fname):
    report_df.iloc[:, 1:] = report_df.iloc[:, 1:].clip(0, 3)
    
    data = [
    go.Heatmap(
        x = report_df.columns.tolist()[1:],
        y = list(map(lambda x: x+1, report_df.index.tolist())),
        z = report_df.iloc[:, 1:].values.tolist(),
        xgap = .1,
        ygap = 1,
        colorscale='Viridis',
        )
    ]

    layout = go.Layout(
        title='Community interests',
        xaxis = dict(title='Categories'),
        yaxis = dict(dtick=1, title='Cluster number' ),
        font = dict(
            size = 16,
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='heatmaps/' + fname + '.html', auto_open=False)

def dump_partition(partition, fname):
    dump_fname = 'partitions/' + fname + '.json'
    
    with open(dump_fname, 'w+') as f:
        json.dump(partition, f)

def calculate_metrics(G, modified_G, vect, names, significance_normalized_vect,
                            mu, sigma, omega, similarity_metric,
                            alpha, gaussian_weighting, fname='foobar.html', algo='louvain', viz=True, avg_by=10, dump_par=False):

    G_w, mixed_weights_stats = create_weighted_graph(modified_G, significance_normalized_vect,
                            mu, sigma, omega, similarity_metric=similarity_metric,
                            alpha=alpha, gaussian_weighting=gaussian_weighting)       

    modularity_list = np.zeros(shape=(avg_by,1))
    modified_modularity_list = np.zeros(shape=(avg_by,1))
    attr_modularity_list = np.zeros(shape=(avg_by,1))
    modularity_density_list = np.zeros(shape=(avg_by,1))
    perm_list = np.zeros(shape=(avg_by,1))
    graph_entropy_list = np.zeros(shape=(avg_by,1))
    tau_list = np.zeros(shape=(avg_by,1))
    cluster_density_list = np.zeros(shape=(avg_by,1))
    cluster_harmony_list = np.zeros(shape=(avg_by,1))
    inter_cluster_density_list = np.zeros(shape=(avg_by,1))
    intra_cluster_density_list = np.zeros(shape=(avg_by,1))

    for i in range(avg_by):
        if algo == 'louvain':
            partition = community.best_partition(G_w)
        elif algo == 'node2vec':
            partition = node2vec_partition(G_w)

        clusters = [[] for i in set(partition.values())]
        for k, v in partition.items():
            clusters[v].append(k)
        
        modularity_list[i] = community.modularity(partition, G, weight='weight')
        modified_modularity_list[i] = community.modularity(partition, G_w, weight='weight')
        modularity_density_list[i] = calc_modularity_density(clusters, G)
        perm_list[i] = permanence(partition, G)
        graph_entropy_list[i] = entropy(G, clusters, vect)
        cluster_density_list[i] = calc_cluster_density(clusters, G)
        cluster_harmony_list[i] = calc_cluster_harmony(clusters, G, vect, similarity_metric, omega)
        inter_cluster_density_list[i] = calc_inter_cluster_density(clusters, G)
        intra_cluster_density_list[i] = calc_intra_cluster_density(clusters, G)

    if dump_par:
        dump_partition(partition, fname)
        
        report_df = generate_report(clusters, vect, names)
        
        tau_list[i], _ = calc_tau(report_df)
    
    if viz:
        vizualize_report(report_df, fname) 
    
    metrics_report = {}
    metrics_report['modularity_mean'] = np.mean(modularity_list)
    metrics_report['modularity_std'] = np.std(modularity_list)
    metrics_report['mod_modularity_mean'] = np.mean(modified_modularity_list)
    metrics_report['mod_modularity_std'] = np.std(modified_modularity_list)
    metrics_report['attr_modularity_mean'] = np.mean(attr_modularity_list)
    metrics_report['attr_modularity_std'] = np.std(attr_modularity_list)    
    metrics_report['permanence_mean'] = np.mean(perm_list)
    metrics_report['permanence_std'] = np.std(perm_list)

    metrics_report['graph_entropy_mean'] = np.mean(graph_entropy_list)
    metrics_report['graph_entropy_std'] = np.std(graph_entropy_list)
    metrics_report['tau_mean'] = np.mean(tau_list)
    metrics_report['tau_std'] = np.std(tau_list)
    metrics_report['modularity_density_mean'] = np.mean(modularity_density_list)
    metrics_report['modularity_density_std'] = np.std(modularity_density_list)

    metrics_report['cluster_density_mean'] = np.mean(cluster_density_list)
    metrics_report['cluster_density_std'] = np.std(cluster_density_list)
    metrics_report['cluster_harmony_mean'] = np.mean(cluster_harmony_list)
    metrics_report['cluster_harmony_std'] = np.std(cluster_harmony_list)

    metrics_report['inter_cluster_density_mean'] = np.mean(inter_cluster_density_list)
    metrics_report['inter_cluster_density_std'] = np.std(inter_cluster_density_list)
    metrics_report['intra_cluster_density_mean'] = np.mean(intra_cluster_density_list)
    metrics_report['intra_cluster_density_std'] = np.std(intra_cluster_density_list)   
    
    return metrics_report, clusters, G, vect, partition

