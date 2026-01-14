import os
import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.image as mpimg
import matplotlib.cm as cm

# -------------------- Utility Functions --------------------

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def read_trajectories(file_path):
    with open(file_path, 'r') as file:
        return {
            cell.strip(): list(map(int, traj.strip().strip('[]').split(',')))
            for cell, traj in (line.strip().split(':') for line in file)
        }

def clean_trajectory(trajectory):
    cleaned, seen = [], set()
    for state in trajectory:
        if state in seen:
            break
        cleaned.append(state)
        seen.add(state)
    return cleaned

def find_attractors(trajectory):
    visited = {}
    for i, state in enumerate(trajectory):
        if state in visited:
            cycle_start = visited[state]
            cycle_states = trajectory[cycle_start:i]
            attractor_state = cycle_states[-1]
            return attractor_state, cycle_states, attractor_state, trajectory[0]
        visited[state] = i
    return trajectory[-1], [], trajectory[-1], trajectory[0]

def analyze_state_visit_frequency(trajectories, state_occupancy_matrix=None, output_csv="state_visit_frequency.csv"):
    """
    Analyze how many times each state has been visited by cells.
    Verifies state occupancy matrix if provided.
    
    Args:
        trajectories: Dictionary of cell trajectories
        state_occupancy_matrix: Optional numpy array (cells x states) to verify
        output_csv: Output CSV filename
    
    Returns:
        DataFrame with state_index, visit_frequency, attractor_frequency
    """
    
    print("\nCalculating visit frequency from trajectories (ground truth)...")
    state_visit_count_traj = {}
    for cell, trajectory in trajectories.items():
        cleaned_traj = clean_trajectory(trajectory)
        for state in cleaned_traj:
            state_visit_count_traj[state] = state_visit_count_traj.get(state, 0) + 1
    
    # If state occupancy matrix is provided, verify it
    if state_occupancy_matrix is not None:
        print(f"\n{'='*60}")
        print("VERIFICATION: Checking state occupancy matrix")
        print(f"{'='*60}")
        print(f"State occupancy matrix shape: {state_occupancy_matrix.shape}")
        print(f"Expected: (num_cells={len(trajectories)}, num_states)")
        
        # Try column sum
        visit_frequency_colsum = state_occupancy_matrix.sum(axis=0)
        print(f"\nColumn sum - min: {visit_frequency_colsum.min()}, max: {visit_frequency_colsum.max()}, total: {visit_frequency_colsum.sum()}")
        
        # Try row sum
        visit_frequency_rowsum = state_occupancy_matrix.sum(axis=1)
        print(f"Row sum - min: {visit_frequency_rowsum.min()}, max: {visit_frequency_rowsum.max()}, total: {visit_frequency_rowsum.sum()}")
        
        # Ground truth
        total_visits_traj = sum(state_visit_count_traj.values())
        print(f"\nGround truth from trajectories: {total_visits_traj} total visits")
        
        # Compare column sum vs ground truth
        print(f"\n--- Testing Column Sum (states as columns) ---")
        state_visit_count_matrix_col = {i+1: int(visit_frequency_colsum[i]) for i in range(len(visit_frequency_colsum))}
        
        matches_col = 0
        mismatches_col = 0
        for state in state_visit_count_traj:
            traj_count = state_visit_count_traj[state]
            matrix_count = state_visit_count_matrix_col.get(state, 0)
            if traj_count == matrix_count:
                matches_col += 1
            else:
                mismatches_col += 1
                if mismatches_col <= 5:  # Show first 5 mismatches
                    print(f"  State {state}: Trajectory={traj_count}, Matrix={matrix_count} ❌")
        
        print(f"\nColumn sum results: {matches_col} matches, {mismatches_col} mismatches")
        if mismatches_col == 0:
            print("COLUMN SUM IS CORRECT! State occupancy matrix is valid.")
        
        print(f"\n--- Testing Row Sum (states as rows) ---")
        print("Note: Row sum would give per-cell state counts, not per-state visit frequency")
        print(f"This doesn't match our needs (we want per-state counts, not per-cell counts)")
        
        print(f"\n{'='*60}")
        print("CONCLUSION:")
        if mismatches_col == 0:
            print("Your state occupancy matrix is CORRECT!")
            print("  - Rows = cells, Columns = states")
            print("  - Column sum gives visit frequency per state")
        else:
            print("State occupancy matrix has mismatches")
            print("  - Check matrix construction logic")
        print(f"{'='*60}\n")
    
    # Use trajectory counts as ground truth
    state_visit_count = state_visit_count_traj
    
    # Count attractor frequency from trajectories
    state_attractor_count = {}
    for cell, trajectory in trajectories.items():
        cleaned_traj = clean_trajectory(trajectory)
        
        # Find attractor for this trajectory
        single_attractor, cyclic_attractor, attractor_state, _ = find_attractors(cleaned_traj)
        attractor = single_attractor if single_attractor is not None else (cyclic_attractor[0] if cyclic_attractor else None)
        
        # Count if this state is an attractor
        if attractor is not None:
            state_attractor_count[attractor] = state_attractor_count.get(attractor, 0) + 1
    
    # Create DataFrame
    all_states = sorted(set(state_visit_count.keys()) | set(state_attractor_count.keys()))
    
    data = {
        'state_index': all_states,
        'visit_frequency': [state_visit_count.get(s, 0) for s in all_states],
        'attractor_frequency': [state_attractor_count.get(s, 0) for s in all_states]
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('visit_frequency', ascending=False)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nState visit frequency saved to {output_csv}")
    print(f"\nTop 10 most visited states:")
    print(df.head(10))
    
    return df

def plot_state_visit_frequency(df, top_n=20):
    """
    Plot state visit frequency as a bar chart.
    
    Args:
        df: DataFrame with state visit frequency data
        top_n: Number of top states to show
    """
    # Get top N states by visit frequency
    df_top = df.head(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Visit frequency
    colors_visit = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top)))
    bars1 = ax1.bar(range(len(df_top)), df_top['visit_frequency'], 
                    color=colors_visit, edgecolor='black', linewidth=2)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(range(len(df_top)))
    ax1.set_xticklabels([f"S{int(s)}" for s in df_top['state_index']], 
                        rotation=45, fontsize=12, ha='right')
    ax1.set_xlabel("State Index", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Visit Frequency (# of Cells)", fontsize=14, fontweight='bold')
    ax1.set_title(f"Top {top_n} Most Visited States", fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Attractor frequency
    df_attractors = df[df['attractor_frequency'] > 0].sort_values('attractor_frequency', ascending=False)
    colors_attr = plt.cm.Reds(np.linspace(0.4, 0.9, len(df_attractors)))
    
    bars2 = ax2.bar(range(len(df_attractors)), df_attractors['attractor_frequency'],
                    color=colors_attr, edgecolor='black', linewidth=2)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(range(len(df_attractors)))
    ax2.set_xticklabels([f"S{int(s)}" for s in df_attractors['state_index']],
                        rotation=45, fontsize=12, ha='right')
    ax2.set_xlabel("State Index (Attractors Only)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Attractor Frequency (# of Cells)", fontsize=14, fontweight='bold')
    ax2.set_title("States Serving as Attractors", fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig("Figure2C1_State_Visit_Frequency.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFigure 2C1 saved as Figure2C1_State_Visit_Frequency.png")

# -------------------- Main Function --------------------
INPUT_FILES = {
    "trajectories": "Combined_Trajectory.txt",
    "state_occupancy": "state_occupancy_matrix.npy",  # Add path to your .npy file
}

def main():
    print("Loading trajectories...")
    trajectories = read_trajectories(INPUT_FILES["trajectories"])
    print(f"Loaded {len(trajectories)} cell trajectories")
    
    # Try to load state occupancy matrix
    state_occupancy_matrix = None
    if os.path.exists(INPUT_FILES["state_occupancy"]):
        print(f"\nLoading state occupancy matrix from {INPUT_FILES['state_occupancy']}...")
        state_occupancy_matrix = np.load(INPUT_FILES["state_occupancy"])
        print(f"State occupancy matrix shape: {state_occupancy_matrix.shape}")
        print(f"Matrix shape: (cells={state_occupancy_matrix.shape[0]}, states={state_occupancy_matrix.shape[1]})")
    else:
        print(f"\nWarning: State occupancy matrix not found at {INPUT_FILES['state_occupancy']}")
        print("Will calculate visit frequency from trajectories instead.")
    
    # Analyze state visit frequency
    print("\nAnalyzing state visit frequency...")
    df = analyze_state_visit_frequency(trajectories, state_occupancy_matrix, output_csv="state_visit_frequency.csv")
    
    # Plot the results
    print("\nGenerating Figure 2C1...")
    plot_state_visit_frequency(df, top_n=20)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total unique states: {len(df)}")
    print(f"Total states serving as attractors: {len(df[df['attractor_frequency'] > 0])}")
    print(f"Most visited state: State {df.iloc[0]['state_index']} ({int(df.iloc[0]['visit_frequency'])} visits)")
    print(f"Most common attractor: State {df[df['attractor_frequency'] > 0].iloc[0]['state_index']} ({int(df[df['attractor_frequency'] > 0].iloc[0]['attractor_frequency'])} cells)")
    
    # Additional analysis: states that are visited but never attractors
    transient_states = df[df['attractor_frequency'] == 0]
    print(f"\nTransient states (visited but never attractors): {len(transient_states)}")
    print(f"Top 5 most visited transient states:")
    print(transient_states.head(5)[['state_index', 'visit_frequency']])
# Run the main function
if __name__ == "__main__":
    main()