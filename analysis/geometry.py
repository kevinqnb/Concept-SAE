"""
Analyze the geometry of the learned feature dictionary.

Examines intra-expert feature similarity (are features within an expert clustered?)
and inter-expert similarity (are experts geometrically distinct from each other?).

Edit CHECKPOINT_DIR below, then run interactively in VS Code or as a script:
    python analysis/geometry.py
"""

# %%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from dictionary_learning.trainers.concept_expert import ConceptExpertAutoEncoder

CHECKPOINT_DIR = "dictionaries/<run_name>"  # edit this
DEVICE = "cpu"

with open(os.path.join(CHECKPOINT_DIR, "config.json")) as f:
    cfg = json.load(f)["trainer"]

ae = ConceptExpertAutoEncoder.from_pretrained(
    os.path.join(CHECKPOINT_DIR, "ae.pt"),
    k=cfg["k"],
    n_experts=cfg["n_experts"],
    num_experts=cfg["num_experts"],
    device=DEVICE,
)
ae.eval()

num_experts     = ae.num_experts
expert_dict_size = ae.expert_dict_size

# Decoder shape: [dict_size, activation_dim]. Rows are unit-norm feature vectors.
# Reshape to [num_experts, expert_dict_size, activation_dim].
decoder = ae.decoder.data.detach()  # [dict_size, activation_dim]
decoder_by_expert = decoder.view(num_experts, expert_dict_size, ae.activation_dim)

print(f"Experts: {num_experts}  |  Features/expert: {expert_dict_size}  |  Activation dim: {ae.activation_dim}")

# %%
# --- 1. Intra-expert cosine similarity ---
# For each expert, compute the full pairwise cosine similarity matrix among its features.
# Since decoder rows are unit-norm, cosine sim = dot product.
# Plot a sample of experts as heatmaps.

NUM_EXPERTS_TO_PLOT = min(6, num_experts)
sample_experts = np.linspace(0, num_experts - 1, NUM_EXPERTS_TO_PLOT, dtype=int)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

intra_mean_cosim = t.zeros(num_experts)

for i, e in enumerate(range(num_experts)):
    W = decoder_by_expert[e]                         # [expert_dict_size, activation_dim]
    cosim = W @ W.T                                  # [expert_dict_size, expert_dict_size]
    # Mask diagonal for the mean (self-similarity = 1 trivially)
    mask = ~t.eye(expert_dict_size, dtype=t.bool)
    intra_mean_cosim[e] = cosim[mask].mean()

    if i < NUM_EXPERTS_TO_PLOT and e in sample_experts:
        ax = axes[list(sample_experts).index(e)]
        im = ax.imshow(cosim.numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"Expert {e}  (mean off-diag cosim: {intra_mean_cosim[e]:.3f})")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")
        plt.colorbar(im, ax=ax)

plt.suptitle("Intra-expert pairwise cosine similarity", fontsize=13)
plt.tight_layout()
plt.savefig("analysis/intra_expert_cosim.png", dpi=150)
plt.show()

# %%
# --- 2. Mean intra-expert cosine similarity per expert ---
# Lower = features within the expert are more diverse / less redundant.

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(num_experts), intra_mean_cosim.numpy())
ax.axhline(intra_mean_cosim.mean().item(), color="red", linestyle="--", label=f"mean = {intra_mean_cosim.mean():.3f}")
ax.set_xlabel("Expert")
ax.set_ylabel("Mean pairwise cosine similarity")
ax.set_title("Intra-expert feature diversity (lower = more diverse)")
ax.legend()
plt.tight_layout()
plt.savefig("analysis/intra_expert_diversity.png", dpi=150)
plt.show()

# %%
# --- 3. Inter-expert cosine similarity ---
# Compare experts by their centroid feature vector (mean over each expert's features).
# Low centroid cosim = experts point in geometrically distinct directions.

centroids = decoder_by_expert.mean(dim=1)                   # [num_experts, activation_dim]
centroids = centroids / centroids.norm(dim=-1, keepdim=True)  # unit-normalize centroids
inter_cosim = centroids @ centroids.T                        # [num_experts, num_experts]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(inter_cosim.numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax.set_xlabel("Expert")
ax.set_ylabel("Expert")
ax.set_title("Inter-expert cosine similarity (centroid vectors)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("analysis/inter_expert_cosim.png", dpi=150)
plt.show()

# Mask diagonal
mask = ~t.eye(num_experts, dtype=t.bool)
print(f"Mean off-diagonal inter-expert cosim: {inter_cosim[mask].mean():.4f}")
print(f"(0 = orthogonal experts, 1 = identical, -1 = opposite)")

# %%
# --- 4. Cross-expert feature similarity ---
# Full pairwise cosim across ALL features (not just centroids).
# Shows whether features cluster by expert or mix across experts.
# Warning: this is [dict_size, dict_size] — may be large for big dictionaries.

if ae.dict_size <= 4096:
    all_cosim = decoder @ decoder.T    # [dict_size, dict_size]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(all_cosim.numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("Feature index (grouped by expert)")
    ax.set_ylabel("Feature index (grouped by expert)")
    ax.set_title("Full pairwise cosine similarity (features sorted by expert)")

    # Draw expert boundary lines
    for i in range(1, num_experts):
        boundary = i * expert_dict_size
        ax.axhline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.5)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("analysis/full_cosim_matrix.png", dpi=150)
    plt.show()
else:
    print(f"Skipping full cosim matrix (dict_size={ae.dict_size} > 4096). Sample a subset manually if needed.")
