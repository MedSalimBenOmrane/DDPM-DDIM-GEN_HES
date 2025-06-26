#!/usr/bin/env python3
"python generate_single_DDPM_fp32.py   --config configs/hes.yml   --checkpoint checkpoints/latest.pt   --batch_size 1   --output_dir sample_256   --timesteps 1000   --sampling_timesteps 1000   --eta 0.0   --seed 42"
""
import os
import argparse
import random
import yaml
import torch
import numpy as np
from torchvision.utils import save_image

from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(
        description="Génère 1 image 256×256 (ou plus) en full‐performance DDPM/DDIM"
    )
    parser.add_argument("--config",             type=str,   default="configs/hes.yml",
                        help="Chemin vers le YAML de config (train_hes.py)")
    parser.add_argument("--checkpoint",         type=str,   required=True,
                        help="Chemin vers latest.pt (model + ema_model)")
    parser.add_argument("--batch_size",         type=int,   default=1,
                        help="Nombre d’images à générer")
    parser.add_argument("--output_dir",         type=str,   default="sample_256",
                        help="Répertoire de sortie pour les PNG")
    parser.add_argument("--timesteps",          type=int,
                        help="Override: nb d’étapes de diffusion (défaut du YAML)")
    parser.add_argument("--sampling_timesteps", type=int,
                        help="Override: nb d’étapes de sampling (DDIM ou DDPM)")
    parser.add_argument("--eta",                type=float, default=0.0,
                        help="ddim_sampling_eta (0=déterministe, 1=stochastique)")
    parser.add_argument("--seed",               type=int,   default=None,
                        help="Graine pour reproductibilité")
    args = parser.parse_args()

    # Fixer la graine si demandé
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[+] Seed fixée → {args.seed}")

    # Charger la config
    cfg = load_config(args.config)

    # Device
    device = "cpu" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Construire le U-Net (identique à l’entraînement)
    model = Unet(
        dim            = cfg['model']['dim'],
        dim_mults      = tuple(cfg['model']['dim_mults']),
        channels       = cfg['model']['in_channels'],
        self_condition = False,
        dropout        = cfg['model']['dropout'],
        flash_attn     = cfg['model'].get('flash_attn', False)
    ).to(device)

    # Choix des étapes
    timesteps_val  = args.timesteps or cfg['diffusion']['timesteps']
    sampling_steps = args.sampling_timesteps or cfg['sampling']['sampling_timesteps']

    # Construire la diffusion (identique à train_hes.py / FID)
    diffusion = GaussianDiffusion(
        model               = model,
        image_size          = cfg['data']['image_size'],
        timesteps           = timesteps_val,
        sampling_timesteps  = sampling_steps,
        objective           = cfg['diffusion']['objective'],
        beta_schedule       = cfg['diffusion']['beta_schedule'],
        ddim_sampling_eta   = args.eta,
        auto_normalize      = True
    ).to(device)
    print(f"[+] Diffusion → timesteps={timesteps_val}, sampling={sampling_steps}, η={args.eta}")

    # Charger checkpoint (EMA si dispo)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('ema_model', ckpt.get('model'))
    diffusion.load_state_dict(state_dict, strict=False)
    diffusion.eval()

    # Génération à partir de bruit pur
    with torch.inference_mode():
        samples = diffusion.sample(batch_size=args.batch_size)
        save_image(
            samples,
            os.path.join(
                args.output_dir,
                "sample.png" if args.batch_size==1 else "samples.png"
            ),
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

    print(f"[+] Génération terminée – voir {args.output_dir}")

if __name__ == "__main__":
    main()
