"DDIM avec 100 step "
"python generate_single_DDIM_fp32.py   --config configs/hes.yml   --checkpoint checkpoints/latest.pt   --batch_size 1   --output_dir sample_256   --timesteps 1000   --seed 400"
#!/usr/bin/env python3
import os
import argparse
import random
import yaml
import torch
import numpy as np
from torchvision.utils import save_image

from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(
        description="Génère 1 image 256×256 à partir de votre checkpoint EMA, en full-precision"
    )
    parser.add_argument("--config",       type=str, default="configs/hes.yml",
                        help="Chemin vers votre fichier de config YAML")
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Chemin vers latest.pt (contenant 'model' et 'ema_model')")
    parser.add_argument("--batch_size",   type=int, default=1,
                        help="Nombre d’images à générer (1 par défaut)")
    parser.add_argument("--output_dir",   type=str, default="sample_256",
                        help="Répertoire de sortie pour votre/vos PNG")
    parser.add_argument("--timesteps",    type=int,
                        help="Nombre d’étapes de diffusion à utiliser (override du YAML)")
    parser.add_argument("--seed",         type=int, default=None,
                        help="Graine aléatoire pour reproductibilité")
    args = parser.parse_args()

    # 0) Fixer la graine si demandé
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[+] Seed fixée → {args.seed}")

    # 1) Charger la config
    cfg = load_config(args.config)

    # 2) Choix du device
    device = "cpu" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 3) Construire le U-Net identique à l’entraînement
    model = Unet(
        dim            = cfg['model']['dim'],
        dim_mults      = tuple(cfg['model']['dim_mults']),
        channels       = cfg['model']['in_channels'],
        self_condition = False,
        dropout        = cfg['model']['dropout'],
        flash_attn     = cfg['model'].get('flash_attn', False)
    ).to(device)

    # 4) Construire le pipeline de diffusion
    diffusion = GaussianDiffusion(
        model               = model,
        image_size          = cfg['data']['image_size'],
        timesteps           = args.timesteps or cfg['diffusion']['timesteps'],
        sampling_timesteps  = cfg['sampling']['sampling_timesteps'],
        objective           = cfg['diffusion']['objective'],
        beta_schedule       = cfg['diffusion']['beta_schedule'],
        ddim_sampling_eta   = 0.0,
        auto_normalize      = True
    ).to(device)
    print(f"[+] Diffusion avec timesteps = {diffusion.num_timesteps}")

    # 5) Charger checkpoint (EMA si disponible)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('ema_model', ckpt.get('model'))
    diffusion.load_state_dict(state_dict, strict=False)
    diffusion.eval()

    # 6) Génération à partir de bruit pur
    with torch.inference_mode():
        samples = diffusion.sample(batch_size=args.batch_size)
        save_image(
            samples,
            os.path.join(args.output_dir, "sample.png"),
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

    print(f"[+] Génération terminée – voir {args.output_dir}/sample.png")

if __name__ == "__main__":
    main()
