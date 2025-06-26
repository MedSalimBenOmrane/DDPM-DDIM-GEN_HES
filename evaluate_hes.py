#!/usr/bin/env python3
"pip install pieapp ou github"
"python evaluate_hes.py \
  --config    configs/hes.yml \
  --checkpoint checkpoints/latest.pt \
  --data_root /home/salim/Desktop/PFE/implimentation/PNP_FM/dataHes \
  --pieapp_weights /chemin/vers/pieapp_weights.pth \
  --out_dir   eval_samples"

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from pieapp.models import PieAPPModel     # ou l'import adapté si vous utilisez un repo local
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from datasets.hes import HESDataset

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_model(cfg, checkpoint, device):
    # Instanciation du U-Net + diffusion
    model = Unet(
        dim            = cfg['model']['dim'],
        dim_mults      = tuple(cfg['model']['dim_mults']),
        channels       = cfg['model']['in_channels'],
        self_condition = False,
        dropout        = cfg['model']['dropout'],
        flash_attn     = cfg['model'].get('flash_attn', False)
    ).to(device)
    diffusion = GaussianDiffusion(
        model               = model,
        image_size          = cfg['data']['image_size'],
        timesteps           = cfg['diffusion']['timesteps'],
        sampling_timesteps  = cfg['sampling']['sampling_timesteps'],
        objective           = cfg['diffusion']['objective'],
        beta_schedule       = cfg['diffusion']['beta_schedule'],
        ddim_sampling_eta   = 0.0,    # reconstruction déterministe
        auto_normalize      = True
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get('ema_model', ckpt.get('model'))
    diffusion.load_state_dict(state_dict, strict=False)
    diffusion.eval()
    return diffusion

def evaluate(args):
    # 1) Préparation
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = build_model(cfg, args.checkpoint, device)

    #PieAPP
    pieapp_net = PieAPPModel()
    pieapp_net.load_state_dict(torch.load(args.pieapp_weights, map_location=device))
    pieapp_net.to(device).eval()

    # Dataset test
    test_ds = HESDataset(root=args.data_root, split="test",
                         image_size=cfg['data']['image_size'],
                         transform=None)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    psnr_list, ssim_list, pieapp_list = [], [], []

    os.makedirs(args.out_dir, exist_ok=True)

    # 2) Boucle d’évaluation
    for i, x0 in enumerate(tqdm(loader, desc="Eval Test")):
        x0 = x0.to(device)  # [1,C,H,W], valeurs [-1,1]

        # bruitage forward équivalent : x_T = q_sample(x0, T-1, eps)
        eps = torch.randn_like(x0)
        t = torch.tensor([diffusion.timesteps-1], device=device)
        xT = diffusion.q_sample(x0, t, noise=eps)

        # reconstruction déterministe (eta=0)
        with torch.no_grad():
            x0_hat = diffusion.sample(
                batch_size=1,
                noise=xT,
                sampling_timesteps=diffusion.timesteps,
                eta=0.0
            )

        # passage CPU & [0,1] pour calcul des métriques
        ref   = ((x0.clamp(-1,1)+1)/2).cpu().numpy().squeeze().transpose(1,2,0)
        rec   = ((x0_hat.clamp(-1,1)+1)/2).cpu().numpy().squeeze().transpose(1,2,0)

        # PSNR / SSIM (skimage attend [0,1])
        psnr_val = compute_psnr(ref, rec, data_range=1.0)
        ssim_val = compute_ssim(ref, rec, data_range=1.0, multichannel=True)

        # PieAPP (torch.Tensor [1,3,H,W] entre 0 et 1)
        # PieAPPModel.forward_pair attend ([B,3,H,W], [B,3,H,W])
        with torch.no_grad():
            pieapp_score = pieapp_net.forward_pair(
                torch.from_numpy(ref.transpose(2,0,1)).unsqueeze(0).to(device),
                torch.from_numpy(rec.transpose(2,0,1)).unsqueeze(0).to(device)
            ).item()

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        pieapp_list.append(pieapp_score)

        # (Optionnel) sauvegarde côte-à-côte
        save_image(
            torch.cat([x0, x0_hat], dim=0),
            os.path.join(args.out_dir, f"{i:04d}_ref_rec.png"),
            nrow=2, normalize=True, value_range=(-1,1)
        )

    # 3) Résultats finaux
    print("=== Évaluation finale sur TEST ===")
    print(f"PSNR   : {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}")
    print(f"SSIM   : {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print(f"PieAPP : {np.mean(pieapp_list):.4f} ± {np.std(pieapp_list):.4f}")
    log_file = 'evaluation_hes_results.txt'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as f:
        f.write(
            f"{timestamp} – "
            f"PSNR: {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}, "
            f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}, "
            f"PieAPP: {np.mean(pieapp_list):.4f} ± {np.std(pieapp_list):.4f}\n"
        )
    print(f"[+] Historique ajouté dans {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, required=True,
                        help="chemin vers configs/hes.yml")
    parser.add_argument("--checkpoint",    type=str, required=True,
                        help="checkpoint final (.pt) à évaluer")
    parser.add_argument("--data_root",     type=str, required=True,
                        help="répertoire racine du dataset (train/val/test)")
    parser.add_argument("--pieapp_weights",type=str, required=True,
                        help="poids PieAPP pré‐entraînés .pth")
    parser.add_argument("--out_dir",       type=str, default="eval_samples",
                        help="où sauver les images comparaison")
    args = parser.parse_args()
    evaluate(args)
