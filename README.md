# A Unified Perspective on Reward Distillation Through Ratio Matching 
## ICML MoFA Workshop 2025

 This repository compares three offline preference-optimization methods (Direct Preference Optimization (DPO), Identity Preference Optimization (IPO), and REBEL) under a unified ratio-matching framework. We derive the REBEL objective as a special case of ratio matching and show its close formal connection to DPO’s implicit reward formulation and IPO’s bounded preference regression. We benchmark all three methods on Anthropic's HH-RLHF dataset. Our code and evaluation routines are provided here to reproduce and extend these experiments.A


## Setup
1. Clone the repo.
```bash
git clone https://github.com/galletas1712/dro.git
cd dro
```

2. Create the environment.
```bash
uv sync
uv venv
source .venv/bin/activate
```

3. Run SFT.
```bash
python sft.py
```

The checkpoints will reside in the ```outputs``` folder.

4. After creating and activating the environment, run our wandb-compatible experiments with Hydra as follows:
```bash
python dpo.py
```

## Hydra Config Details
Some key configuration settings include: 

- `model.name`  
  Path to the base model checkpoint used for DPO (e.g. `/root/dro/outputs/.../epoch-0-step-119999`).

- `model.tokenizer`  
  Hugging Face tokenizer ID (e.g. `EleutherAI/pythia-2.8b`).

- `dataset.name`  
  Dataset with `prompt`/`chosen`/`rejected` columns for training and eval (e.g. `claserken/hhrlhf-rejudged`).

- `dpo.beta`  
  Divergence control parameter for DPO; higher values enforce closer adherence to the reference model (default: `5.0`).

- `dpo.loss_type`  
  Preference‐optimization loss to use: `"rebel"`, `"sigmoid"`(DPO), `"hinge"`, or `"ipo"`.

- `training.learning_rate`  
  Base learning rate for DPO.

  - `wandb.run_name`  
  Template for W&B run naming, parameterized by loss type and β.

## Metrics
We primarily use reward margin as our evaluation metric because it directly measures the difference in model‐predicted reward between chosen and rejected responses, providing a quantitative measure of alignment under our ratio‐matching framework.  