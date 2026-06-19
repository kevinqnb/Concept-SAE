import os
import tempfile

import pytest
import torch as t

from dictionary_learning.trainers.concept_expert import ConceptExpertAutoEncoder, ConceptExpertTrainer

ACTIVATION_DIM = 16
NUM_EXPERTS = 8
EXPERT_DICT_SIZE = 8
DICT_SIZE = NUM_EXPERTS * EXPERT_DICT_SIZE  # 64
K = 2
N_EXPERTS = 3
BATCH = 32


def reference_decode(top_acts, top_indices, decoder, b_dec):
    """Pure-PyTorch decode equivalent to TritonDecoder; runs on CPU."""
    selected = decoder[top_indices]
    weighted = (selected * top_acts.unsqueeze(-1)).sum(1)
    return weighted + b_dec


@pytest.fixture(scope="module")
def ae():
    model = ConceptExpertAutoEncoder(ACTIVATION_DIM, DICT_SIZE, K, N_EXPERTS, NUM_EXPERTS)
    model.eval()
    return model


@pytest.fixture(scope="module")
def sample_x():
    t.manual_seed(0)
    return t.randn(BATCH, ACTIVATION_DIM)


@pytest.fixture(scope="module")
def encoded_f(ae, sample_x):
    with t.no_grad():
        return ae.encode(sample_x)


@pytest.fixture(scope="module")
def router_output(ae, sample_x):
    with t.no_grad():
        p = t.softmax(ae.router(sample_x - ae.b_router), dim=-1)
        _, top_expert_idx = p.topk(N_EXPERTS, dim=-1)
    return p, top_expert_idx


# --- 1. Initialization ---

def test_init_decoder_unit_norm(ae):
    norms = ae.decoder.data.norm(dim=1)
    assert t.allclose(norms, t.ones_like(norms), atol=1e-5)


def test_init_expert_dict_size(ae):
    assert ae.expert_dict_size == EXPERT_DICT_SIZE


def test_init_rejects_indivisible_dict_size():
    with pytest.raises(AssertionError):
        ConceptExpertAutoEncoder(16, 65, 2, 3, 8)


def test_init_rejects_n_experts_gt_num_experts():
    with pytest.raises(AssertionError):
        ConceptExpertAutoEncoder(16, 64, 2, 10, 8)


# --- 2. Encode: shape and sparsity ---

def test_encode_shape(encoded_f):
    assert encoded_f.shape == (BATCH, DICT_SIZE)


def test_encode_max_sparsity(encoded_f):
    nonzeros = (encoded_f != 0).sum(dim=-1)
    assert (nonzeros <= N_EXPERTS * K).all()


def test_encode_nonnegative(encoded_f):
    assert (encoded_f >= 0).all()


# --- 3. Encode: nonzeros fall in the correct expert slices ---

def test_encode_nonzeros_in_selected_expert_slices(encoded_f, router_output):
    _, top_expert_idx = router_output
    for b in range(BATCH):
        nonzero_positions = encoded_f[b].nonzero(as_tuple=True)[0]
        selected = set(top_expert_idx[b].tolist())
        for pos in nonzero_positions:
            assert pos.item() // EXPERT_DICT_SIZE in selected


# --- 4. Encode: values = router_prob * encoder_topk ---

def test_encode_values_equal_router_prob_times_topk(ae, sample_x, encoded_f, router_output):
    p, top_expert_idx = router_output
    with t.no_grad():
        z = t.relu(ae.encoder(sample_x - ae.b_dec))
        z_by_expert = z.view(BATCH, NUM_EXPERTS, EXPERT_DICT_SIZE)

    for b in range(BATCH):
        selected_experts = top_expert_idx[b].tolist()
        expert_probs = p[b, selected_experts]

        for i, e in enumerate(selected_experts):
            start = e * EXPERT_DICT_SIZE
            actual_slice = encoded_f[b, start:start + EXPERT_DICT_SIZE]
            z_e = z_by_expert[b, e]
            topk_vals, topk_local = z_e.topk(K)
            expected_slice = t.zeros(EXPERT_DICT_SIZE)
            expected_slice[topk_local] = topk_vals * expert_probs[i]
            assert t.allclose(actual_slice, expected_slice, atol=1e-5)

        non_selected = set(range(NUM_EXPERTS)) - set(selected_experts)
        for e in non_selected:
            start = e * EXPERT_DICT_SIZE
            assert (encoded_f[b, start:start + EXPERT_DICT_SIZE] == 0).all()


# --- 5. Reconstruction = weighted sum of expert outputs ---

def test_reconstruction_is_weighted_sum_of_expert_outputs(ae):
    x1 = t.randn(1, ACTIVATION_DIM)
    with t.no_grad():
        f1 = ae.encode(x1)
        top_vals, top_idx = f1.topk(N_EXPERTS * K, sorted=False)
        x_hat_decode = reference_decode(top_vals, top_idx, ae.decoder.data, ae.b_dec)

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

    assert t.allclose(x_hat_decode, x_hat_manual.unsqueeze(0), atol=1e-5)


# --- 6. Unit-norm constraint ---

def test_set_decoder_norm_to_unit_norm():
    ae_local = ConceptExpertAutoEncoder(ACTIVATION_DIM, DICT_SIZE, K, N_EXPERTS, NUM_EXPERTS)
    with t.no_grad():
        ae_local.decoder.data *= 3.7
    ae_local.set_decoder_norm_to_unit_norm()
    norms = ae_local.decoder.data.norm(dim=1)
    assert t.allclose(norms, t.ones_like(norms), atol=1e-5)


# --- 7. Save and reload ---

def test_save_and_reload(ae):
    ae.set_decoder_norm_to_unit_norm()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ae.pt")
        t.save(ae.state_dict(), path)
        ae2 = ConceptExpertAutoEncoder.from_pretrained(path, k=K, n_experts=N_EXPERTS, num_experts=NUM_EXPERTS)

        for (n1, p1), (n2, p2) in zip(ae.named_parameters(), ae2.named_parameters()):
            assert t.allclose(p1, p2), f"Weight mismatch in {n1}"

        x_test = t.randn(4, ACTIVATION_DIM)
        assert t.allclose(ae.encode(x_test), ae2.encode(x_test))


# --- 8. Trainer (requires CUDA) ---

@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
def test_trainer_loss_and_update():
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

    losses = [trainer.update(step, x_cuda) for step in range(5)]
    assert all(isinstance(l, float) for l in losses)

    norms = trainer.ae.decoder.data.norm(dim=1)
    assert t.allclose(norms, t.ones_like(norms), atol=1e-4)

    trainer.ae.zero_grad()
    loss = trainer.loss(x_cuda, step=0)
    loss.backward()
    for name, param in trainer.ae.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
