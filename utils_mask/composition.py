import os
from collections import defaultdict
from plyfile import PlyData
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

import ipywidgets as widgets
from IPython.display import display, clear_output

from PLY_utils import compute_collisons
import argparse


def print_highest_pairs_until_zero(matrix):
    """
    Given a symmetric collision score matrix {obj1:{obj2:score}},
    repeatedly:
        - find the highest-scoring pair
        - print it
        - remove both objects from further consideration
    Stops when all remaining scores are zero or < 2 objects remain.
    """

    # Work on a local copy
    active = set(matrix.keys())

    # Flatten all pairs into a list
    def get_best_pair():
        best_pair = None
        best_score = 0

        for a in active:
            for b in active:
                if a >= b:  # avoid duplicates + diagonal
                    continue
                score = matrix[a].get(b, 0)
                if score > best_score:
                    best_score = score
                    best_pair = (a, b, score)

        return best_pair  # either (a,b,score) or None

    results = []

    while True:

        if len(active) < 2:
            break

        best = get_best_pair()

        if best is None or best[2] == 0:
            # No more positive-scoring pairs
            break

        a, b, score = best
        results.append(best)

        print(f"Selected pair: ({fetch_label(a)}, {fetch_label(b)}) -> score {score:.4f}")

        # Remove both from the active set
        active.remove(a)
        active.remove(b)

    print("\nFinished. Final selected pairs:")
    for a, b, s in results:
        print(f"({fetch_label(a)}, {fetch_label(b)}) -> {s:.4f}")

    return results

def compute_collision_matrices(ply_path):
    """
    Compute pairwise collision scores for all PLY files in ply_path.
    Each score is the number of colliding points divided by the smaller point cloud size.
    Returns a nested dict {label_i: {label_j: score}}.
    """
    # list only files that look like PLYs (skip dirs and other files)
    point_clouds = [f for f in os.listdir(ply_path)
                    if os.path.isfile(os.path.join(ply_path, f)) and f.lower().endswith('.ply')]
    collision_matrix = defaultdict(dict)
    for i in range(len(point_clouds)):
       
        for j in range(i+1,len(point_clouds)):

            label_i = point_clouds[i]
            label_j = point_clouds[j]

            ply_i_path = os.path.join(ply_path, label_i)
            ply_j_path = os.path.join(ply_path, label_j)
            # skip if any entry is not a regular file (safety)
            if not os.path.isfile(ply_i_path) or not os.path.isfile(ply_j_path):
                continue

            ply_i = PlyData.read(ply_i_path)
            ply_j = PlyData.read(ply_j_path)
            collisions = compute_collisons(ply_i, ply_j)
        
            label_i = label_i.replace(".ply", "")
            label_j = label_j.replace(".ply", "")
            #print(f"{label_i}-{label_j}: {collisions}")

            vertices_i = ply_i['vertex'].count
            vertices_j = ply_j['vertex'].count

            
            if(label_i == label_j): collisions = 0
            collision_matrix[label_i][label_j] = collisions/min(vertices_i, vertices_j)
            collision_matrix[label_j][label_i] = collisions/min(vertices_i, vertices_j)
            
    return collision_matrix



def print_collision_matrix(matrix):
    """Print the collision score matrix as a pandas DataFrame."""
    df = pd.DataFrame(matrix).fillna(0) 
    print(df)


def plot_collision_matrix_heatmap(matrix):
    """Plot a heatmap of pairwise collision scores and print the top 10 colliding pairs."""
    labels = list(matrix.keys())
    obj_labels = [fetch_label(key) for key in labels]
    size = len(labels)

   
        
    data = np.array([[matrix[i].get(j, 0) for j in labels] for i in labels])

    data_sum = []
    for i in range(size):
        data_sum.append(sum(data[i]))
    
    normalized_data =  np.array([[data[i,j]/data_sum[i] for j in range(size)] for i in range(size)])

    fig, ax = plt.subplots()


    # Extract upper triangle indices (excluding diagonal)
    rows, cols = np.triu_indices_from(normalized_data, k=1)

    # Pair each (i, j) with its collision value
    collisions = [((i, j), data[i, j]) for i, j in zip(rows, cols)]

    # Sort by collision value (descending)
    collisions_sorted = sorted(collisions, key=lambda x: x[1], reverse=True)

    # Print top N pairs with highest collisions
    top_n = 10
    for idx, ((i, j), val) in enumerate(collisions_sorted[:top_n], start=1):
        l_1 = labels[i]
        l_2 = labels[j]
        print(f"{idx}. Pair ({fetch_label(l_1)}, {fetch_label(l_2)}) - Collision: {val}")
   
    cax = ax.imshow(data, cmap='viridis')


    fig.colorbar(cax)

    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels(obj_labels)
    ax.set_yticklabels(obj_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    for i in range(size):
        for j in range(size):

            ax.text(j, i, "{:.2f}".format(data[i, j]), ha='center', va='center', color='r')

    ax.set_title("Collision Matrix")
    plt.tight_layout()
    plt.show()

def compute_PLY_size(ply_path):
    """Print a size-sorted dict mapping each PLY filename to its vertex count."""
    point_clouds = os.listdir(ply_path)
    size_dict = defaultdict(dict)
    for i in range(len(point_clouds)):
        label_i = point_clouds[i]
        ply_i = PlyData.read(os.path.join(ply_path, label_i))
        vertices_i = ply_i['vertex'].count
        size_dict[label_i] = vertices_i
    sorted_dict = {k: v for k, v in sorted(size_dict.items(), key=lambda item: item[1])}
    print(sorted_dict)

       
       

def plot_collision_matrix_heatmap_interactive(matrix):
    """Interactively display the collision heatmap and let the user remove objects by index until satisfied."""
    labels = list(matrix.keys())

    while True:
        obj_labels = [fetch_label(key) for key in labels]
        size = len(labels)

        data = np.array([[matrix[i].get(j, 0) for j in labels] for i in labels])
        
        # Compute normalized data (if needed later)
        data_sum = [sum(row) for row in data]
        normalized_data = np.array([
            [data[i, j] / data_sum[i] if data_sum[i] != 0 else 0 for j in range(size)]
            for i in range(size)
        ])

        # Show top N collisions
        rows, cols = np.triu_indices_from(normalized_data, k=1)
        collisions = [((i, j), data[i, j]) for i, j in zip(rows, cols)]
        collisions_sorted = sorted(collisions, key=lambda x: x[1], reverse=True)

        print("\nTop collisions:")
        top_n = 10
        for idx, ((i, j), val) in enumerate(collisions_sorted[:top_n], start=1):
            l_1, l_2 = labels[i], labels[j]
            print(f"{idx}. Pair ({fetch_label(l_1)}, {fetch_label(l_2)}) - Collision: {val}")

        # Plot the heatmap
        fig, ax = plt.subplots()
        cax = ax.imshow(data, cmap='viridis')
        fig.colorbar(cax)

        ax.set_xticks(np.arange(size))
        ax.set_yticks(np.arange(size))
        ax.set_xticklabels(obj_labels)
        ax.set_yticklabels(obj_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(size):
            for j in range(size):
                ax.text(j, i, "{:.2f}".format(data[i, j]), ha='center', va='center', color='r')

        ax.set_title("Collision Matrix")
        plt.tight_layout()
        plt.show()

        # Prompt user to remove a label
        print("\nObject Labels:")
        for i, label in enumerate(obj_labels):
            print(f"{i}: {label}")

        to_remove = input("\nEnter index(es) of object(s) to remove, separated by commas (or 'q' to quit): ")

        if to_remove.lower() == 'q':
            break

        try:
            indices = [int(x.strip()) for x in to_remove.split(',')]
            labels = [label for i, label in enumerate(labels) if i not in indices]
        except Exception as e:
            print("Invalid input. Please enter indices separated by commas.")

        if len(labels) < 2:
            print("Not enough data left to display a heatmap. Exiting.")
            break

def compose_ply(ply_a, ply_b):
    """Concatenate two PLY vertex arrays."""
    return ply_a + ply_b



def compose_mask(mask_a, mask_b):
    """Alpha-composite two RGBA PIL mask images, returning the merged result."""
    image = Image.alpha_composite(mask_a, mask_b)
    return image


def fetch_label(name):
    """Strip the .ply extension and return the bare label string."""
    name = name.replace(".ply","")
    labels = name.split("_")

    label =""
    n = len(labels)
 
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a ScanNet scene")
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name, e.g. scene0200_00"
    )
    args = parser.parse_args()

    SCENE = args.scene
    
    PLY_PATH = os.path.join("./output",SCENE, "tmp")
    cm = compute_collision_matrices(PLY_PATH) 
    #print_highest_pairs_until_zero(cm)
