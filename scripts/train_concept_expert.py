import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnsight import LanguageModel
import torch as t
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, cfg_filename
from dictionary_learning.trainers.concept_expert import ConceptExpertAutoEncoder, ConceptExpertTrainer
from dictionary_learning.evaluation import evaluate
import wandb
import argparse
import itertools
from config import lm, activation_dim, layer, hf, steps, n_ctxs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True)
parser.add_argument('--dict_ratio', type=int, default=32)
parser.add_argument("--ks", nargs="+", type=int, required=True)
parser.add_argument("--num_experts", nargs="+", type=int, required=True)
parser.add_argument("--n_experts", nargs="+", type=int, required=True)
args = parser.parse_args()

device = f'cuda:{args.gpu}'
model = LanguageModel(lm, dispatch=True, device_map=device)
submodule = model.transformer.h[layer]
data = hf_dataset_to_generator(hf)
buffer = ActivationBuffer(data, model, submodule, d_submodule=activation_dim, n_ctxs=n_ctxs, device=device)

base_trainer_config = {
    'trainer': ConceptExpertTrainer,
    'dict_class': ConceptExpertAutoEncoder,
    'activation_dim': activation_dim,
    'dict_size': args.dict_ratio * activation_dim,
    'decay_start': int(steps * 0.8),
    'steps': steps,
    'seed': 0,
    'device': device,
    'layer': layer,
    'lm_name': lm,
    'wandb_name': 'ConceptExpertAutoEncoder',
}

trainer_configs = [
    base_trainer_config | {'k': combo[0], 'num_experts': combo[1], 'n_experts': combo[2]}
    for combo in itertools.product(args.ks, args.num_experts, args.n_experts)
]

wandb.init(
    entity="amudide",
    project="Concept Expert SAE",
    config={f'{tc["wandb_name"]}-{i}': tc for i, tc in enumerate(trainer_configs)},
)

trainSAE(buffer, trainer_configs=trainer_configs, save_dir='dictionaries', log_steps=1, steps=steps)

print("Training finished. Evaluating SAE...", flush=True)
for i, trainer_config in enumerate(trainer_configs):
    ae = ConceptExpertAutoEncoder.from_pretrained(
        f'dictionaries/{cfg_filename(trainer_config)}/ae.pt',
        k=trainer_config['k'],
        n_experts=trainer_config['n_experts'],
        num_experts=trainer_config['num_experts'],
        device=device,
    )
    metrics = evaluate(ae, buffer, device=device)
    log = {f'{trainer_config["wandb_name"]}-{i}/{k}': v for k, v in metrics.items()}
    wandb.log(log, step=steps + 1)

wandb.finish()
