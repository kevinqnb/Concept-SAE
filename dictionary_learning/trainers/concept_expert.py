import einops
import torch as t
import torch.nn as nn
from collections import namedtuple

from ..dictionary import Dictionary
from ..trainers.trainer import SAETrainer


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median of `points`. Used for initializing decoder bias."""
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess
        weights = 1 / t.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if t.norm(guess - prev) < tol:
            break

    return guess


class ConceptExpertAutoEncoder(Dictionary, nn.Module):
    """
    Concept Expert SAE: routes each token to the top-N experts and sums their reconstructions.

    Each expert owns a contiguous slice of the dictionary of size (dict_size // num_experts).
    For each selected expert, the top-k features within its slice are activated and weighted
    by the expert's router probability. The final reconstruction is the sum of all N experts'
    weighted partial reconstructions.

    Total active features per token: N * k.
    """
    def __init__(self, activation_dim, dict_size, k, n_experts, num_experts):
        """
        k          : features selected per expert
        n_experts  : number of experts to route to (top-N)
        num_experts: total number of experts in the dictionary
        """
        super().__init__()
        assert dict_size % num_experts == 0, "dict_size must be divisible by num_experts"
        assert n_experts <= num_experts, "n_experts cannot exceed num_experts"

        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        self.n_experts = n_experts
        self.num_experts = num_experts
        self.expert_dict_size = dict_size // num_experts

        self.encoder = nn.Linear(activation_dim, dict_size, bias=False)
        self.router = nn.Linear(activation_dim, num_experts, bias=False)

        self.decoder = nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))
        self.b_router = nn.Parameter(t.zeros(activation_dim))

    def encode(self, x):
        """
        x : [batch, activation_dim]
        Returns a sparse feature vector of shape [batch, dict_size] with exactly
        n_experts * k nonzero entries per token, weighted by router probabilities.
        """
        batch = x.shape[0]

        z = nn.functional.relu(self.encoder(x - self.b_dec))          # [batch, dict_size]
        p = t.softmax(self.router(x - self.b_router), dim=-1)          # [batch, num_experts]

        # Select top-N experts per token
        top_expert_vals, top_expert_idx = p.topk(self.n_experts, dim=-1)  # [batch, n_experts]

        # Reshape encoder output by expert, then gather selected experts
        z_by_expert = z.view(batch, self.num_experts, self.expert_dict_size)
        expert_features = z_by_expert.gather(
            1,
            top_expert_idx.unsqueeze(-1).expand(-1, -1, self.expert_dict_size)
        )  # [batch, n_experts, expert_dict_size]

        # Top-k within each selected expert's slice
        topk_vals, topk_local_idx = expert_features.topk(self.k, dim=-1)  # [batch, n_experts, k]

        # Weight activations by router probability
        topk_vals = topk_vals * top_expert_vals.unsqueeze(-1)             # [batch, n_experts, k]

        # Convert local per-expert indices to global dictionary indices
        expert_offsets = (top_expert_idx * self.expert_dict_size).unsqueeze(-1)  # [batch, n_experts, 1]
        topk_global_idx = topk_local_idx + expert_offsets                        # [batch, n_experts, k]

        # Scatter into a full sparse feature vector
        f = t.zeros(batch, self.dict_size, device=x.device, dtype=x.dtype)
        f.scatter_(1, topk_global_idx.view(batch, -1), topk_vals.view(batch, -1))

        return f

    def decode(self, top_acts, top_indices):
        from ..kernels import TritonDecoder
        return TritonDecoder.apply(top_indices, top_acts, self.decoder.mT) + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x.view(-1, x.shape[-1]))
        top_acts, top_indices = f.topk(self.n_experts * self.k, sorted=False)
        x_hat = self.decode(top_acts, top_indices).view(x.shape)
        f = f.view(*x.shape[:-1], f.shape[-1])
        if not output_features:
            return x_hat
        elif output_features == "all":
            return x_hat, f, top_acts, top_indices
        else:
            return x_hat, f

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder.dtype).eps
        norm = t.norm(self.decoder.data, dim=1, keepdim=True)
        self.decoder.data /= norm + eps

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.grad is not None
        parallel_component = einops.einsum(
            self.decoder.grad,
            self.decoder.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.decoder.grad -= einops.einsum(
            parallel_component,
            self.decoder.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def from_pretrained(path, k=100, n_experts=2, num_experts=16, device=None):
        state_dict = t.load(path, map_location=device)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = ConceptExpertAutoEncoder(activation_dim, dict_size, k, n_experts, num_experts)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class ConceptExpertTrainer(SAETrainer):
    """
    Trainer for ConceptExpertAutoEncoder.
    Load-balancing loss is omitted; loss is reconstruction (L2) only.
    """
    def __init__(self,
                 dict_class=ConceptExpertAutoEncoder,
                 activation_dim=512,
                 dict_size=384*64,
                 k=4,
                 n_experts=16,
                 num_experts=384,
                 decay_start=24000,
                 steps=30000,
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='ConceptExpertAutoEncoder',
                 submodule_name=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.activation_dim = activation_dim
        self.wandb_name = wandb_name
        self.steps = steps
        self.k = k
        self.n_experts = n_experts
        self.num_experts = num_experts

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.ae = dict_class(activation_dim, dict_size, k, n_experts, num_experts)
        self.device = device if device is not None else ('cuda' if t.cuda.is_available() else 'cpu')
        self.ae.to(self.device)

        scale = dict_size / (2 ** 14)
        self.lr = 2e-4 / scale ** 0.5

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        def lr_fn(step):
            if step < decay_start:
                return 1.
            else:
                return (steps - step) / (steps - decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)

        self.logging_parameters = ["effective_l0", "dead_features"]
        self.effective_l0 = -1
        self.dead_features = -1

    def loss(self, x, step=None, logging=False):
        x = x.to(self.device)
        total_active = self.n_experts * self.k

        f = self.ae.encode(x)
        top_acts, top_indices = f.topk(total_active, sorted=False)
        x_hat = self.ae.decode(top_acts, top_indices)

        e = x_hat - x

        self.effective_l0 = total_active

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        self.dead_features = int((self.num_tokens_since_fired > 10_000_000).sum())

        l2_loss = e.pow(2).sum(dim=-1).mean()

        if not logging:
            return l2_loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {'l2_loss': l2_loss.item(), 'loss': l2_loss.item()}
            )

    def update(self, step, x):
        x = x.to(self.device)

        if step == 0:
            median = geometric_median(x)
            self.ae.b_dec.data = median
            self.ae.b_router.data = median

        self.ae.set_decoder_norm_to_unit_norm()

        loss = self.loss(x, step=step)
        loss.backward()

        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.ae.remove_gradient_parallel_to_decoder_directions()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return loss.item()

    @property
    def config(self):
        return {
            'trainer_class': 'ConceptExpertTrainer',
            'dict_class': 'ConceptExpertAutoEncoder',
            'lr': self.lr,
            'steps': self.steps,
            'seed': self.seed,
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'k': self.ae.k,
            'n_experts': self.ae.n_experts,
            'num_experts': self.ae.num_experts,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }
