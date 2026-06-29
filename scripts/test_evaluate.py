"""
Quick smoke test for the evaluate() / loss_recovered() code path.
Skips training entirely — uses a randomly initialized AE.
Run on CPU (default) or pass --gpu 0 to use a GPU.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.trainers.concept_expert import ConceptExpertAutoEncoder
from dictionary_learning.evaluation import evaluate
from config import cfg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=None)
args = parser.parse_args()
device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

# Small AE parameters — just enough to satisfy the constraints
k = 4
num_experts = 16
n_experts = 4
dict_size = cfg.activation_dim * 4   # 4x ratio, divisible by num_experts

assert dict_size % num_experts == 0

print(f"Device: {device}")
print(f"Loading {cfg.lm}...")
model = LanguageModel(cfg.lm, dispatch=True, device_map=device)
submodule = model.transformer.h[cfg.layer]
data = hf_dataset_to_generator(cfg.hf)

# Small buffer: 128 contexts × 128 tokens = 16k activations
buffer = ActivationBuffer(
    data, model, submodule,
    d_submodule=cfg.activation_dim,
    n_ctxs=128,
    out_batch_size=512,
    device=device,
)

ae = ConceptExpertAutoEncoder(
    activation_dim=cfg.activation_dim,
    dict_size=dict_size,
    k=k,
    n_experts=n_experts,
    num_experts=num_experts,
).to(device)

print("Running evaluate() (this exercises loss_recovered)...")
metrics = evaluate(ae, buffer, max_len=64, batch_size=8, device=device)
print("Success!")
for k_, v in metrics.items():
    print(f"  {k_}: {v:.4f}")
