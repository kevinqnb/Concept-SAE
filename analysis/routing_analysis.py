"""
Analyze expert routing behavior from a cached routing file.
Run cache_routing.py first to generate data/routing_cache.pt.

Run interactively in VS Code or as a script:
    python analysis/routing_analysis.py
"""

# %%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as t
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
from config import lm

CACHE_PATH = "data/routing_cache.pt"

cache = t.load(CACHE_PATH, map_location="cpu")
expert_idx   = cache["expert_idx"]    # [num_tokens, n_experts]
expert_probs = cache["expert_probs"]  # [num_tokens, n_experts]
token_ids    = cache["token_ids"]     # [num_tokens]
cfg          = cache["config"]

num_tokens  = token_ids.shape[0]
num_experts = cfg["num_experts"]
n_experts   = cfg["n_experts"]

print(f"Tokens: {num_tokens:,}  |  Total experts: {num_experts}  |  Selected per token: {n_experts}")

tokenizer = AutoTokenizer.from_pretrained(lm)

# %%
# --- 1. Expert load distribution ---
# Count how many times each expert appears across all token-expert selections.

expert_counts = t.zeros(num_experts, dtype=t.long)
expert_counts.scatter_add_(0, expert_idx.view(-1), t.ones(num_tokens * n_experts, dtype=t.long))

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(num_experts), expert_counts.numpy())
ax.axhline(num_tokens * n_experts / num_experts, color="red", linestyle="--", label="uniform")
ax.set_xlabel("Expert")
ax.set_ylabel("Tokens routed")
ax.set_title("Expert load distribution")
ax.legend()
plt.tight_layout()
plt.savefig("analysis/expert_load.png", dpi=150)
plt.show()
print("Expert load — min:", expert_counts.min().item(), "max:", expert_counts.max().item())

# %%
# --- 2. Top tokens per expert ---
# For each expert, find the most common tokens routed to it.

# Build a boolean assignment matrix: assigned[token, expert] = True
assigned = t.zeros(num_tokens, num_experts, dtype=t.bool)
assigned.scatter_(1, expert_idx, t.ones_like(expert_idx, dtype=t.bool))

print("\nTop-20 tokens per expert:")
for e in range(num_experts):
    tokens_for_expert = token_ids[assigned[:, e]].tolist()
    counts = Counter(tokens_for_expert).most_common(20)
    decoded = [(tokenizer.decode([tok]).strip(), cnt) for tok, cnt in counts]
    print(f"\n  Expert {e:>3d} ({expert_counts[e].item():>7,} tokens):")
    print("  " + "  |  ".join(f"'{tok}' ({cnt})" for tok, cnt in decoded[:10]))

# %%
# --- 3. Expert co-occurrence matrix ---
# For each pair of experts, how often are they both selected for the same token?

cooccurrence = t.zeros(num_experts, num_experts, dtype=t.long)
for i in range(n_experts):
    for j in range(n_experts):
        if i == j:
            continue
        pairs = expert_idx[:, i] * num_experts + expert_idx[:, j]
        for p in pairs.tolist():
            cooccurrence[p // num_experts, p % num_experts] += 1

# Normalize by total tokens
cooccurrence_norm = cooccurrence.float() / num_tokens

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cooccurrence_norm.numpy(), aspect="auto")
ax.set_xlabel("Expert j")
ax.set_ylabel("Expert i")
ax.set_title("Expert co-occurrence (fraction of tokens)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("analysis/expert_cooccurrence.png", dpi=150)
plt.show()

# %%
# --- 4. Router probability distribution by rank ---
# How do softmax probabilities differ between the 1st vs 2nd vs ... Nth selected expert?
# (expert_probs are already sorted descending by topk)

fig, ax = plt.subplots(figsize=(7, 4))
for rank in range(n_experts):
    probs_at_rank = expert_probs[:, rank].numpy()
    ax.hist(probs_at_rank, bins=50, alpha=0.6, label=f"Rank {rank + 1}")
ax.set_xlabel("Router probability")
ax.set_ylabel("Count")
ax.set_title("Router probability distribution by selection rank")
ax.legend()
plt.tight_layout()
plt.savefig("analysis/router_prob_distribution.png", dpi=150)
plt.show()

# %%
# --- 5. Per-token routing entropy ---
# Low entropy = router is confident (one expert dominates).
# High entropy = router spreads probability evenly across selected experts.

# Compute entropy over the full softmax distribution using all expert probs
# We only have the top-N probs, so compute entropy over those (approximate).
probs_clipped = expert_probs / expert_probs.sum(dim=-1, keepdim=True)  # renormalize top-N
entropy = -(probs_clipped * (probs_clipped + 1e-9).log()).sum(dim=-1)  # [num_tokens]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(entropy.numpy(), bins=60)
ax.set_xlabel("Entropy (nats, over top-N experts)")
ax.set_ylabel("Count")
ax.set_title("Per-token routing entropy")
plt.tight_layout()
plt.savefig("analysis/routing_entropy.png", dpi=150)
plt.show()
print(f"Mean entropy: {entropy.mean():.3f}  |  Max possible: {np.log(n_experts):.3f}")
