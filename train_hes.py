# train_hes.py

import os
import argparse
import logging
import yaml
from itertools import cycle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
from accelerate import Accelerator

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import EMA
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from datasets.hes import HESDataset  # ajustez si nécessaire

def make_transforms(cfg):
    tf = [T.Resize(cfg['data']['image_size'])]
    if cfg['data']['random_flip']:
        tf.append(T.RandomHorizontalFlip())
    tf += [
        T.CenterCrop(cfg['data']['image_size']),
        T.ToTensor(),  # pixels en [0,1]
    ]
    return T.Compose(tf)

def main():
    # ─── Arguments & config ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/hes.yml')
    parser.add_argument('--resume', action='store_true',
                        help="Relancer depuis le dernier checkpoint si existant")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger()

    # ─── Dossiers ───────────────────────────────────────────────────────────
    checkpoint_dir = cfg['training'].get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    # on crée aussi le dossier pour les stats FID
    stats_dir = cfg['training'].get('fid_stats_dir', checkpoint_dir)
    os.makedirs(stats_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
    loss_log_path  = os.path.join(checkpoint_dir, 'loss_log.txt')

    # Ouvrir le fichier de log en mode append
    loss_log_file = open(loss_log_path, 'a')
    fid_log_path = os.path.join(checkpoint_dir, 'fid_log.txt')
    fid_log_file = open(fid_log_path, 'a')

    # ─── Accelerator (FP16 + accumulation) ─────────────────────────────────
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=cfg['training'].get('gradient_accumulate_every', 1)
    )

    # ─── Device-agnostic : on passera tout dans `accelerator.prepare` ──────
    # 1️⃣ Dataset & DataLoader
    transforms = make_transforms(cfg)
    train_ds = HESDataset(cfg['data']['root'], split='train', transform=transforms)
    val_ds   = HESDataset(cfg['data']['root'], split='val',   transform=transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    # 2️⃣ Modèle & diffusion
    model = Unet(
        dim=cfg['model']['dim'],
        dim_mults=tuple(cfg['model']['dim_mults']),
        channels=cfg['model']['in_channels'],
        self_condition=False,
        dropout=cfg['model']['dropout'],
        flash_attn=cfg['model'].get('flash_attn', False)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=cfg['data']['image_size'],
        timesteps=cfg['diffusion']['timesteps'],
        sampling_timesteps=cfg['sampling']['sampling_timesteps'],
        objective=cfg['diffusion']['objective'],
        beta_schedule=cfg['diffusion']['beta_schedule'],
        ddim_sampling_eta=0.0,
        auto_normalize=True
    )

    optimizer = torch.optim.Adam(
        diffusion.parameters(),
        lr=cfg['optim']['lr'],
        betas=(cfg['optim']['beta1'], 0.999),
        eps=cfg['optim']['eps']
    )

    ema = EMA(diffusion, beta=cfg['model']['ema_rate'])

    # 3️⃣ Préparation par accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    device = accelerator.device
    diffusion = diffusion.to(device)
    ema.ema_model = ema.ema_model.to(device)
    # 4️⃣ FID-evaluator
    val_iter = cycle(val_loader)
    fid_evaluator = FIDEvaluation(
        batch_size=cfg['training']['batch_size'],
        dl=val_iter,
        sampler=ema.ema_model,
        accelerator=accelerator,
        stats_dir=stats_dir,
        device=device,
        num_fid_samples=cfg['training'].get('num_fid_samples', len(val_ds)),
        inception_block_idx=2048
    )

    # 5️⃣ Chargement checkpoint éventuel
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.ema_model.load_state_dict(ckpt['ema_model'])
        if 'scaler' in ckpt and accelerator.scaler is not None:
            accelerator.scaler.load_state_dict(ckpt['scaler'])
        global_step = ckpt['step']
        start_epoch = ckpt['epoch'] + 1
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}, step {global_step}")

    # ─── Boucle d’entraînement ───────────────────────────────────────────────
    n_epochs   = cfg['training']['n_epochs']
    max_steps  = cfg['training']['n_iters']
    snap_freq  = cfg['training']['snapshot_freq']
    val_freq   = cfg['training']['validation_freq']
    grad_clip  = cfg['optim'].get('grad_clip', 1.0)

    for epoch in range(start_epoch, n_epochs):
        logger.info(f"Epoch {epoch+1}/{n_epochs} – steps done: {global_step}/{max_steps}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)

        for batch in pbar:
            imgs = batch[0].to(device)

            with accelerator.autocast():
                loss = diffusion(imgs)
            accelerator.backward(loss)

            # gradient step only when accumulation terminée
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                ema.update()
                global_step += 1

                # —— 1) écrire dans tqdm
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

                # —— 2) écrire dans le fichier de log
                loss_log_file.write(
                    f"Epoch: {epoch+1}, iter: {global_step}, Loss: {loss.item():.4f}\n"
                )
                loss_log_file.flush()

                # ③ snapshot & sample
                if global_step % snap_freq == 0:
                    ema.ema_model.eval()
                    with torch.inference_mode():
                        samples = ema.ema_model.sample(batch_size=cfg['sampling']['batch_size'])
                    save_image(
                        samples,
                        os.path.join(checkpoint_dir, f"sample_{global_step}.png"),
                        nrow=4,
                        normalize=True,
                        value_range=(-1, 1)
                    )
                    ema.ema_model.train()

                # ④ validation & FID
                if global_step % val_freq == 0:
                    fid = fid_evaluator.fid_score()
                    accelerator.print(f"[Step {global_step}] FID: {fid:.4f}")
                    # Écrire dans le fichier fid_log.txt
                    fid_log_file.write(
                        f"Epoch: {epoch+1}, iter: {global_step}, FID: {fid:.4f}\n"
                    )
                    fid_log_file.flush()
                # ⑤ checkpointing
                if accelerator.is_local_main_process and global_step % snap_freq == 0:
                    ckpt = {
                        'model':       accelerator.get_state_dict(model),
                        'optimizer':   optimizer.state_dict(),
                        'ema_model':   ema.ema_model.state_dict(),
                        'scaler':      accelerator.scaler.state_dict() if accelerator.scaler else None,
                        'epoch':       epoch,
                        'step':        global_step
                    }
                    torch.save(ckpt, checkpoint_path)

            # arrêt si on atteint le budget d’itérations
            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    # fermer le fichier de log
    loss_log_file.close()
    fid_log_file.close()
    logger.info("=== Entraînement terminé ===")

if __name__ == '__main__':
    main()
