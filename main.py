"""
Structural Causal Modeling (SCM) & Extremal Tail Dependence
Author: Ikechukwu Okechi Kamalu
Repository: https://github.com/ikechukwukamalu8/extremal-causal-learning
Description: Demonstrates causal discovery limitations on heavy-tailed 
             Pareto distributions and calculates tail-dependence metrics.
"""

import sys
import subprocess

# Auto-install required packages if missing
def install_requirements():
    required_packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'networkx', 'gcastle']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from castle.algorithms import PC

# Set random seed for reproducibility
np.random.seed(42)

def generate_heavy_tailed_scm(n_samples=8000, tail_index=2.0):
    """Generates synthetic heavy-tailed SCM data with Pareto noise."""
    X = (np.random.pareto(a=tail_index, size=n_samples) + 1) * 5.0
    Y = 0.8 * X + (np.random.pareto(a=tail_index, size=n_samples) + 1) * 2.0
    Z = 0.4 * Y + 0.3 * X + np.random.exponential(scale=3.0, size=n_samples)
    
    return pd.DataFrame({'X_Rainfall': X, 'Y_RiverFlow': Y, 'Z_Groundwater': Z})

def compute_extremal_chi(x, y, q=0.95):
    """Computes empirical tail dependence coefficient Chi_q."""
    threshold_x = np.quantile(x, q)
    threshold_y = np.quantile(y, q)
    x_extreme = x > threshold_x
    y_extreme = y > threshold_y
    return np.sum(x_extreme & y_extreme) / np.sum(x_extreme)

def draw_directed_dag(adj_matrix, labels, title, ax):
    """Renders directed graphs with explicit node margins and arrowheads."""
    G = nx.DiGraph()
    for label in labels:
        G.add_node(label)
        
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1:
                G.add_edge(labels[i], labels[j])

    pos = {'X_Rainfall': (0, 1), 'Y_RiverFlow': (1, 0), 'Z_Groundwater': (2, 1)}
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(
        G, pos, 
        arrowstyle='->', 
        arrowsize=20, 
        edge_color='navy', 
        width=2, 
        connectionstyle='arc3,rad=0.1',
        min_source_margin=25,
        min_target_margin=25,
        ax=ax
    )
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

def main():
    print("==================================================")
    print("1. GENERATING HEAVY-TAILED DATASET")
    print("==================================================")
    df = generate_heavy_tailed_scm()
    feature_names = df.columns.tolist()
    
    true_dag = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ])
    
    print("\n==================================================")
    print("2. EMPIRICAL TAIL DEPENDENCE COEFFICIENTS (Chi at q=0.95)")
    print("==================================================")
    chi_xy = compute_extremal_chi(df['X_Rainfall'], df['Y_RiverFlow'], 0.95)
    chi_yz = compute_extremal_chi(df['Y_RiverFlow'], df['Z_Groundwater'], 0.95)
    chi_xz = compute_extremal_chi(df['X_Rainfall'], df['Z_Groundwater'], 0.95)
    
    print(f"Chi(X -> Y): {chi_xy:.4f}")
    print(f"Chi(Y -> Z): {chi_yz:.4f}")
    print(f"Chi(X -> Z): {chi_xz:.4f}")
    
    print("\n==================================================")
    print("3. RUNNING STRUCTURE LEARNING (PC ALGORITHM)")
    print("==================================================")
    q95_X = np.quantile(df['X_Rainfall'], 0.95)
    df_tail = df[df['X_Rainfall'] > q95_X]
    
    pc_full = PC()
    pc_full.learn(df.values)
    
    pc_tail = PC()
    pc_tail.learn(df_tail.values)
    
    print("\n==================================================")
    print("4. RENDERING & SAVING FIGURE")
    print("==================================================")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    draw_directed_dag(true_dag, feature_names, "Ground Truth DAG\n(True Causal Directions)", axes[0])
    draw_directed_dag(pc_full.causal_matrix, feature_names, "Learned DAG: PC Algorithm\n(Full Dataset)", axes[1])
    draw_directed_dag(pc_tail.causal_matrix, feature_names, "Learned DAG: PC Algorithm\n(Extreme Tail Subset Q95)", axes[2])
    
    plt.tight_layout()
    plt.savefig("dag_comparison.png", dpi=300, bbox_inches='tight')
    print("Plot successfully saved as 'dag_comparison.png'.")
    plt.show()

if __name__ == "__main__":
    main()
