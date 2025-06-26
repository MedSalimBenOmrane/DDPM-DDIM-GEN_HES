#!/usr/bin/env python3
"python count_params.py --config configs/hes.yml"
import argparse
import yaml
from denoising_diffusion_pytorch import Unet

def load_config(path: str) -> dict:
    """Charge le YAML de config et le retourne sous forme de dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_unet(cfg: dict) -> Unet:
    """Construit et retourne uniquement le modèle U-Net (sans wrapper diffusion)."""
    return Unet(
        dim            = cfg['model']['dim'],
        dim_mults      = tuple(cfg['model']['dim_mults']),
        channels       = cfg['model']['in_channels'],
        self_condition = False,
        dropout        = cfg['model']['dropout'],
        flash_attn     = cfg['model'].get('flash_attn', False)
    )

def count_parameters(model):
    """Retourne (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compte les paramètres de votre U-Net configuré pour DDPM"
    )
    parser.add_argument(
        "--config", type=str, default="configs/hes.yml",
        help="Chemin vers le YAML de config (ex: configs/hes.yml)"
    )
    args = parser.parse_args()

    # 1) Charger la config
    cfg = load_config(args.config)

    # 2) Instancier le modèle
    model = build_unet(cfg)

    # 3) Compter les paramètres
    total, trainable = count_parameters(model)

    # 4) Affichage
    print("=== Statistiques du modèle U-Net ===")
    print(f"Total parameters      : {total:,}")
    print(f"Trainable parameters  : {trainable:,}")
    print(f"Non-trainable params  : {total - trainable:,}")
