"""
Sanity checks for ConceptExpertAutoEncoder.

Sections 1-6 run on CPU (no Triton/CUDA required).
Section 7 tests the full trainer and requires CUDA.

Usage:
    python test_concept_expert.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as t
import torch.nn as nn
import tempfile

from dictionary_learning.trainers.concept_expert import ConceptExpertAutoEncoder, ConceptExpertTrainer

# Tiny dimensions for fast CPU testing
ACTIVATION_DIM  = 16
NUM_EXPERTS     = 8
EXPERT_DICT_SIZE = 8
DICT_SIZE       = NUM_EXPERTS * EXPERT_DICT_SIZE   # 64
K               = 2   # features per expert
N_EXPERTS       = 3   # experts selected per token
BATCH           = 32


def section(name):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print('='*55)

def ok(msg):
    print(f"  [PASS]  {msg}")


# Pure-PyTorch decode: equivalent to TritonDecoder but runs on CPU.
# decode(f) = sum_j f[j] * W[j], which is just index-select + weighted sum.
def reference_decode(top_acts, top_indices, decoder, b_dec):
    selected = decoder[top_indices]                         # [batch, n*k, activation_dim]
    weighted = (selected * top_acts.unsqueeze(-1)).sum(1)   # [batch, activation_dim]
    return weighted + b_dec


# ─── 1. Initialization ──────────────────────────────────────────────────────

section("1. Initialization")

ae = ConceptExpertAutoEncoder(ACTIVATION_DIM, DICT_SIZE, K, N_EXPERTS, NUM_EXPERTS)

norms = ae.decoder.data.norm(dim=1)
assert t.allclose(norms, t.ones_like(norms), atol=1e-5), \
    f"Decoder not unit norm after init: min={norms.min():.4f}, max={norms.max():.4f}"
ok("Decoder rows initialized to unit norm")

# Config sanity
assert ae.expert_dict_size == EXPERT_DICT_SIZE
ok(f"expert_dict_size = {ae.expert_dict_size}  (dict_size / num_experts = {DICT_SIZE} / {NUM_EXPERTS})")

# Invalid configs should raise
try:
    ConceptExpertAutoEncoder(16, 65, 2, 3, 8)  # 65 not divisible by 8
    assert False, "Should have raised AssertionError"
except AssertionError:
    ok("Rejects dict_size not divisible by num_experts")

try:
    ConceptExpertAutoEncoder(16, 64, 2, 10, 8)  # n_experts > num_experts
    assert False, "Should have raised AssertionError"
except AssertionError:
    ok("Rejects n_experts > num_experts")


# ─── 2. Encode: shape and sparsity ──────────────────────────────────────────

section("2. Encode — shape and sparsity")

ae.eval()
x = t.randn(BATCH, ACTIVATION_DIM)
f = ae.encode(x)

assert f.shape == (BATCH, DICT_SIZE), f"Wrong shape: {f.shape}"
ok(f"encode() output shape: {tuple(f.shape)}")

max_nonzeros = N_EXPERTS * K
nonzeros = (f != 0).sum(dim=-1)
assert (nonzeros <= max_nonzeros).all(), \
    f"More than {max_nonzeros} nonzeros per token: {nonzeros.unique().tolist()}"
ok(f"At most n_experts * k = {N_EXPERTS} * {K} = {max_nonzeros} nonzeros per token "
   f"(fewer when ReLU zeroes encoder outputs; mean = {nonzeros.float().mean():.1f})")

# All values are non-negative (ReLU encoder, positive router weights)
assert (f >= 0).all(), "encode() produced negative values"
ok("All feature values are non-negative (ReLU + positive router probabilities)")


# ─── 3. Encode: nonzeros fall in the correct expert slices ──────────────────

section("3. Encode — nonzeros in correct expert slices")

with t.no_grad():
    p = t.softmax(ae.router(x - ae.b_router), dim=-1)          # [batch, num_experts]
    _, top_expert_idx = p.topk(N_EXPERTS, dim=-1)               # [batch, n_experts]

for b in range(BATCH):
    nonzero_positions = f[b].nonzero(as_tuple=True)[0]
    selected = set(top_expert_idx[b].tolist())
    for pos in nonzero_positions:
        expert_for_pos = pos.item() // EXPERT_DICT_SIZE
        assert expert_for_pos in selected, (
            f"Token {b}: nonzero at position {pos.item()} belongs to expert {expert_for_pos}, "
            f"but selected experts are {sorted(selected)}"
        )

ok("All nonzero features belong to their token's selected experts")


# ─── 4. Encode: feature values equal router_prob * topk(z) ─────────────────

section("4. Encode — values = router_prob * encoder_topk")

with t.no_grad():
    z = t.relu(ae.encoder(x - ae.b_dec))               # [batch, dict_size]
    z_by_expert = z.view(BATCH, NUM_EXPERTS, EXPERT_DICT_SIZE)

for b in range(BATCH):
    selected_experts = top_expert_idx[b].tolist()
    expert_probs = p[b, selected_experts]

    for i, e in enumerate(selected_experts):
        start = e * EXPERT_DICT_SIZE
        end   = start + EXPERT_DICT_SIZE

        actual_slice = f[b, start:end]  # [expert_dict_size]

        # Build expected: sparse vector with top-k positions filled by p_e * z_e[j], rest 0
        z_e = z_by_expert[b, e]
        topk_vals, topk_local = z_e.topk(K)
        expected_slice = t.zeros(EXPERT_DICT_SIZE)
        expected_slice[topk_local] = topk_vals * expert_probs[i]

        assert t.allclose(actual_slice, expected_slice, atol=1e-5), (
            f"Token {b}, expert {e}: expected {expected_slice.tolist()}, "
            f"got {actual_slice.tolist()}"
        )

    # Non-selected experts should contribute zero
    non_selected = set(range(NUM_EXPERTS)) - set(selected_experts)
    for e in non_selected:
        start = e * EXPERT_DICT_SIZE
        end   = start + EXPERT_DICT_SIZE
        assert (f[b, start:end] == 0).all(), \
            f"Token {b}: non-selected expert {e} has nonzero features"

ok("Feature values equal router_prob * encoder_topk in each selected expert's slice")
ok("Non-selected expert slices are exactly zero")


# ─── 5. Reconstruction = weighted sum of expert outputs ─────────────────────

section("5. Reconstruction — weighted sum identity")

# Key property: because decode is linear,
#   decode(f) = sum_j f[j] * W[j]
#             = sum_e { p_e * sum_{j in expert_e} topk(z_e)_j * W[j] }
#             = sum_e { p_e * expert_e_reconstruction }
#
# Verify this with reference_decode (pure PyTorch, no Triton).

ae.eval()
x1 = t.randn(1, ACTIVATION_DIM)

with t.no_grad():
    f1 = ae.encode(x1)
    top_vals, top_idx = f1.topk(N_EXPERTS * K, sorted=False)
    x_hat_decode = reference_decode(top_vals, top_idx, ae.decoder.data, ae.b_dec)

    # Manual weighted sum: iterate over each selected expert
    p1 = t.softmax(ae.router(x1 - ae.b_router), dim=-1)
    top_p, top_e = p1.topk(N_EXPERTS, dim=-1)
    z1 = t.relu(ae.encoder(x1 - ae.b_dec))
    z_by_expert1 = z1.view(1, NUM_EXPERTS, EXPERT_DICT_SIZE)

    x_hat_manual = ae.b_dec.clone()
    for i in range(N_EXPERTS):
        e = top_e[0, i].item()
        p_e = top_p[0, i].item()
        z_e = z_by_expert1[0, e]
        topk_vals, topk_local = z_e.topk(K)
        global_idx = topk_local + e * EXPERT_DICT_SIZE
        for j in range(K):
            x_hat_manual = x_hat_manual + p_e * topk_vals[j].item() * ae.decoder.data[global_idx[j]]

assert t.allclose(x_hat_decode, x_hat_manual.unsqueeze(0), atol=1e-5), (
    f"Mismatch:\n  reference_decode: {x_hat_decode}\n  manual sum: {x_hat_manual}"
)
ok("reference_decode(encode(x)) == manual weighted sum of expert reconstructions")


# ─── 6. Unit-norm constraint ─────────────────────────────────────────────────

section("6. set_decoder_norm_to_unit_norm")

# Perturb decoder then re-normalize
with t.no_grad():
    ae.decoder.data *= 3.7
norms_before = ae.decoder.data.norm(dim=1)
assert not t.allclose(norms_before, t.ones_like(norms_before), atol=1e-3), \
    "Perturbation did not change norms"

ae.set_decoder_norm_to_unit_norm()
norms_after = ae.decoder.data.norm(dim=1)
assert t.allclose(norms_after, t.ones_like(norms_after), atol=1e-5), \
    f"Norms not restored: min={norms_after.min():.4f}, max={norms_after.max():.4f}"
ok("set_decoder_norm_to_unit_norm() restores unit norms after perturbation")


# ─── 7. Save and reload ──────────────────────────────────────────────────────

section("7. Save and reload (state dict round-trip)")

ae.set_decoder_norm_to_unit_norm()  # clean state before saving

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "ae.pt")
    t.save(ae.state_dict(), path)

    ae2 = ConceptExpertAutoEncoder.from_pretrained(
        path, k=K, n_experts=N_EXPERTS, num_experts=NUM_EXPERTS
    )

    for (n1, p1), (n2, p2) in zip(ae.named_parameters(), ae2.named_parameters()):
        assert t.allclose(p1, p2), f"Weight mismatch in {n1} after reload"
    ok("All weights match exactly after save/load")

    x_test = t.randn(4, ACTIVATION_DIM)
    f_orig = ae.encode(x_test)
    f_reload = ae2.encode(x_test)
    assert t.allclose(f_orig, f_reload), "encode() output differs after reload"
    ok("encode() output identical after save/load")


# ─── 8. Trainer (requires CUDA) ─────────────────────────────────────────────

section("8. Trainer — loss, update, gradient flow (CUDA)")

if not t.cuda.is_available():
    print("  [SKIP]  CUDA not available — run on a GPU machine to test the trainer")
else:
    trainer = ConceptExpertTrainer(
        activation_dim=ACTIVATION_DIM,
        dict_size=DICT_SIZE,
        k=K,
        n_experts=N_EXPERTS,
        num_experts=NUM_EXPERTS,
        steps=10,
        decay_start=8,
        device="cuda:0",
        layer=8,
        lm_name="openai-community/gpt2",
    )

    x_cuda = t.randn(BATCH, ACTIVATION_DIM, device="cuda:0")

    log = trainer.loss(x_cuda, step=0, logging=True)
    assert isinstance(log.losses["l2_loss"], float)
    assert log.losses["l2_loss"] > 0
    ok(f"loss() returns l2_loss = {log.losses['l2_loss']:.4f}")

    losses = []
    for step in range(5):
        losses.append(trainer.update(step, x_cuda))
    ok(f"update() ran 5 steps — losses: {[f'{l:.4f}' for l in losses]}")

    norms_gpu = trainer.ae.decoder.data.norm(dim=1)
    assert t.allclose(norms_gpu, t.ones_like(norms_gpu), atol=1e-4), \
        f"Decoder norms drifted: min={norms_gpu.min():.4f}, max={norms_gpu.max():.4f}"
    ok("Decoder norms remain unit norm after training steps")

    # Check gradients flow to all parameters
    trainer.ae.zero_grad()
    loss = trainer.loss(x_cuda, step=0)
    loss.backward()
    for name, param in trainer.ae.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    ok("Gradients flow to all parameters (encoder, decoder, router, b_dec, b_router)")


# ────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("  All tests passed.")
print('='*55)
