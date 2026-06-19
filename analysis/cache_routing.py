"""
Run a trained ConceptExpertAutoEncoder over cached activations and save
per-token expert routing decisions to disk.

Output: data/routing_cache.pt containing:
    expert_idx   [num_tokens, n_experts]  - which experts were selected
    expert_probs [num_tokens, n_experts]  - corresponding router probabilities
    token_ids    [num_tokens]             - token id for each position

Usage:
    python analysis/cache_routing.py --checkpoint_dir dictionaries/<run_name>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch as t
from tqdm import tqdm
from dictionary_learning.trainers.concept_expert import ConceptExpertAutoEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", required=True, help="Path to a saved SAE directory (must contain ae.pt and config.json)")
parser.add_argument("--activations", default="data/activations_layer8.pt", help="Path to cached activations tensor")
parser.add_argument("--tokens", default="data/tokens.pt", help="Path to cached token ids tensor")
parser.add_argument("--output", default="data/routing_cache.pt", help="Where to save the routing cache")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

# Load SAE from checkpoint, reading hyperparams from the saved config
with open(os.path.join(args.checkpoint_dir, "config.json")) as f:
    cfg = json.load(f)["trainer"]

ae = ConceptExpertAutoEncoder.from_pretrained(
    os.path.join(args.checkpoint_dir, "ae.pt"),
    k=cfg["k"],
    n_experts=cfg["n_experts"],
    num_experts=cfg["num_experts"],
    device=args.device,
)
ae.eval()

# Load and flatten activations: [num_contexts, ctx_len, d] -> [num_tokens, d]
print("Loading activations...", flush=True)
activations = t.load(args.activations, map_location="cpu")   # [num_contexts, ctx_len, d]
token_ids = t.load(args.tokens, map_location="cpu")          # [num_contexts, ctx_len]

num_contexts, ctx_len, d = activations.shape
activations = activations.view(-1, d)   # [num_tokens, d]
token_ids = token_ids.view(-1)          # [num_tokens]
num_tokens = activations.shape[0]

print(f"Tokens: {num_tokens:,}  |  Experts: {cfg['num_experts']}  |  N selected: {cfg['n_experts']}")

all_expert_idx = []
all_expert_probs = []

with t.no_grad():
    for start in tqdm(range(0, num_tokens, args.batch_size), desc="Caching routing"):
        x = activations[start : start + args.batch_size].to(args.device)
        p = t.softmax(ae.router(x - ae.b_router), dim=-1)          # [batch, num_experts]
        probs, idx = p.topk(ae.n_experts, dim=-1)                   # [batch, n_experts]
        all_expert_idx.append(idx.cpu())
        all_expert_probs.append(probs.cpu())

routing_cache = {
    "expert_idx":   t.cat(all_expert_idx,   dim=0),   # [num_tokens, n_experts]
    "expert_probs": t.cat(all_expert_probs, dim=0),   # [num_tokens, n_experts]
    "token_ids":    token_ids,                         # [num_tokens]
    "config":       cfg,
}

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
t.save(routing_cache, args.output)
print(f"Saved routing cache to {args.output}")
