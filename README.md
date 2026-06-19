# Concept Expert SAE

A mechanistic interpretability research codebase for training and analyzing **Concept Expert Sparse Autoencoders (SAEs)**.

The core architecture routes each token to the top-N experts, selects the top-k features within each expert's slice of the dictionary, and reconstructs the activation as a probability-weighted sum of the N expert reconstructions. The hypothesis is that experts specialize in distinct concept clusters, making the learned features more interpretable than a flat dictionary.

Built on top of [dictionary_learning](https://github.com/saprmarks/dictionary_learning) by Samuel Marks and Aaron Mueller. Cloned from [switch_sae](https://github.com/amudide/switch_sae/tree/main) by Mudide et al. 

---

## Architecture

Each token's activation `x` is reconstructed as:

```
x_hat = sum_i( p_i · decode(topk(encode(x)[expert_i])) )
```

where `p_i` is the softmax router probability for expert `i`, and the sum is over the N selected experts. Total active features per token: **N × k**.

**Parameters:**
- `num_experts` — total number of experts partitioning the dictionary
- `n_experts` (N) — how many experts are selected per token (top-N routing)
- `k` — features activated per expert

All models are trained on GPT-2 (`openai-community/gpt2`) layer 8 activations extracted from OpenWebText.

**Default configuration:** `num_experts=384`, `n_experts=16`, `k=4`, `dict_size=24576`.

The default dictionary size of 24576 = 384 × 64 is set to match the standard 32× expansion used in the TopK SAE literature (32 × 768 activation dim = 24576). This allows direct comparison with baseline results at the same dictionary size. Each expert owns 64 features, and with 16 experts selected per token and k=4 features active per expert, 64 features fire per token in total — the same total sparsity as a flat TopK SAE with k=64. The routing sparsity is 16/384 ≈ 4%, meaning each token activates a small fraction of the concept vocabulary.

---

## Setup

```bash
uv sync
```

Requires CUDA for training. Global config (model, dataset, training steps) lives in `config.py`.

---

## Training

All scripts are run from the repo root.

**Main architecture:**
```bash
python scripts/train_concept_expert.py --gpu 0 --ks 4 --num_experts 384 --n_experts 16
```

**Baselines:**
```bash
# Top-1 Switch SAE (original)
python scripts/train_switch.py --gpu 0 --ks 64 --num_experts 16 --lb_alphas 3.0

# Switch with one expert always active (1-on variant)
python scripts/train_switch_1on.py --gpu 0 --ks 64 --num_experts 16 --lb_alphas 3.0

# Standard TopK SAE
python scripts/train_topk.py --gpu 0 --ks 64
```

Trained SAEs are saved to `dictionaries/<run_name>/ae.pt` alongside a `config.json` recording all hyperparameters.

Multiple hyperparameter combinations can be swept in a single run by passing multiple values:
```bash
python scripts/train_concept_expert.py --gpu 0 --ks 4 8 --num_experts 384 --n_experts 8 16
```

All runs log to Weights & Biases.

---

## Evaluation

### 1. Cache activations

```bash
python scripts/save_activations.py
# → data/activations_layer8.pt  [num_contexts, ctx_len, 768]
# → data/tokens.pt              [num_contexts, ctx_len]
```

### 2. Cache routing decisions

```bash
python analysis/cache_routing.py --checkpoint_dir dictionaries/<run_name>
# → data/routing_cache.pt
```

### 3. Analyze expert routing

```bash
python analysis/routing_analysis.py
```

Produces plots in `analysis/`:
- **Expert load** — how uniformly tokens are distributed across experts
- **Top tokens per expert** — which tokens each expert specializes in
- **Co-occurrence matrix** — which pairs of experts are selected together
- **Router probability by rank** — how confident the router is in its top-N selections
- **Routing entropy** — per-token uncertainty in expert assignment

### 4. Analyze dictionary geometry

Edit `CHECKPOINT_DIR` at the top of `analysis/geometry.py`, then:

```bash
python analysis/geometry.py
```

Produces plots in `analysis/`:
- **Intra-expert cosine similarity** — are features within an expert diverse or redundant?
- **Inter-expert centroid similarity** — are experts geometrically distinct from each other?
- **Full feature cosine similarity matrix** — do features cluster by expert? (block-diagonal = yes)

---
