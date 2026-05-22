################################################################################
# Split&Splat - Copyright (c) 2026, MEDIALab, University of Padova.
#
# Author(s):
#  Leonardo Monchieri (leonardo.monchieri@unipd.it)
#  Elena Camuffo (elenacamuffo97@gmail.com)
#  Francesco Barbato (francesco.barbato@dei.unipd.it)
#  Pietro Zanuttigh (zanuttigh@dei.unipd.it)
#  Simone Milani (simone.milani@dei.unipd.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################

from scene.cameras import Camera

import torch
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import json

from sklearn.decomposition import PCA


def count_samples(img_name):
    """
    Counts files inside ./testing_view/<DATASET>/black_bg/<image_name>/
    If folder doesn't exist, return 0.
    """
    folder = os.path.join("./testing_view", DATASET, "black_bg", img_name)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

# -------------------------------
# Load your saved descriptors
# -------------------------------

DATASET = "waldo_kitchen"
image_descriptors = torch.load(os.path.join("./data", DATASET, "clip_descriptors.pt"), weights_only=False)
label_descriptors = torch.load(os.path.join("./data", DATASET, "all_text_descriptors.pt"), weights_only=False)

# Convert to lists for consistent order
image_names = list(image_descriptors.keys())
label_names = list(label_descriptors.keys())

# Stack tensors into matrices
image_embs = torch.stack([image_descriptors[n] for n in image_names])
label_embs = torch.stack([label_descriptors[n] for n in label_names])

# Normalize (should already be normalized, but just in case)
image_embs = F.normalize(image_embs, dim=1)
label_embs = F.normalize(label_embs, dim=1)


mean_embs = image_embs.mean()

#image_embs = image_embs - mean_embs


label_embs = label_embs.squeeze(1)

# -------------------------------
# Compute cosine similarity matrix
# -------------------------------
similarity_matrix = label_embs  @ image_embs.mT  # (num_images x num_labels)

# make sure it’s 2D
similarity_matrix = similarity_matrix.squeeze()


# Choose PCA output dimension (e.g., 128, 256, or preserve 95% variance)
PCA_DIM = 128

# Combine both image + text embeddings so PCA learns joint space
all_embs = torch.cat([image_embs, label_embs], dim=0)

max_pca_dim = min(all_embs.shape[0], all_embs.shape[1])
PCA_DIM = min(30, max_pca_dim)  # or any number <= max_pca_dim
print("Using PCA dim:", PCA_DIM)

# Fit PCA in numpy
pca = PCA(n_components=PCA_DIM)
all_embs_pca = pca.fit_transform(all_embs.numpy())

# Split projected embeddings back
image_embs_pca = torch.tensor(all_embs_pca[:len(image_embs)], dtype=torch.float32)
label_embs_pca = torch.tensor(all_embs_pca[len(image_embs):], dtype=torch.float32)

# Re-normalize after PCA (important!)
image_embs = F.normalize(image_embs_pca, dim=1)
label_embs = F.normalize(label_embs_pca, dim=1)

similarity_matrix = label_embs @ image_embs.mT
similarity_matrix = similarity_matrix.squeeze()
# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame(similarity_matrix.numpy(), index=label_names, columns=image_names)


# Optional: Save to CSV for analysis
df.to_csv("object_label_correlation.csv")

# -------------------------------
# Print top matches per object
# -------------------------------
for obj in label_names:
    sims = df.loc[obj].sort_values(ascending=False)
    top_images = sims.head(3).index.tolist()
    print(f"\n🔹 {obj} — Top 3 matches:")
    for label, score in sims.head(3).items():
        print(f"   {label:<30} {score:.5f}")



best_matches = {}
best_scores = {}

for lbl in label_names:
    
    sims = df.loc[lbl]                      # Get similarity scores for this label
    
    best_img = sims.idxmax()                # Find the image with the highest similarity
    best_score = sims.max().item()
   
    best_matches[lbl] = best_img             # Store in dictionary

best_matches_img = {}

for img in image_names:
    
    sims = df[img]                          # Get similarity scores for this label
   
    best_label = sims.idxmax()              # Find the image with the highest similarity
    best_score = sims.max().item()
    
    best_matches_img[img] = best_label      # Store in dictionary

print(best_matches_img)


# # Optionally save to JSON
# with open(os.path.join("./data", DATASET, "label_best_matches.json"), "w") as f:
#     json.dump(best_matches, f, indent=4)

best_matches = {}

DELTA = 1e-3                                    # adjust if needed (1e-3 for looser matching)

for lbl in label_names:
    sims = df.loc[lbl]

    max_val = sims.max()                        # max correlation score

    # collect all images whose similarity is "close enough" to maximum
    tied_images = sims[abs(sims - max_val) <= DELTA].index.tolist()

    if len(tied_images) == 1:
      
        best_matches[lbl] = tied_images[0]     # No tie — this is the best match
    else:
        # Tie — pick the one with most samples
        sample_counts = {img: count_samples(img) for img in tied_images}
        best_img = max(sample_counts, key=sample_counts.get)

        best_matches[lbl] = best_img

        print(f"[Delta tie resolved] {lbl}: {tied_images} → kept {best_img} "
              f"(samples={sample_counts[best_img]}, delta={DELTA})")



# # Save results
with open(os.path.join("./data", DATASET, "label_best_matches.json"), "w") as f:
    json.dump(best_matches, f, indent=4)

print("\n✔ label_best_matches.json saved.")