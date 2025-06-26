# DDPM-DDIM-GEN_HES

# Génération d’images histopathologiques avec DDPM / DDIM

## Description
Ce dépôt implémente un modèle de diffusion (DDPM et DDIM) pour la génération d’images histopathologiques (HES) à partir de bruit gaussien.  
Il contient :
- le code d’entraînement  
- les scripts d’échantillonnage (DDPM & DDIM)  
- les utilitaires pour l’évaluation et le suivi des métriques (loss, FID, PSNR, SSIM, pieAPP, …)  
- un dossier de configurations, d’assets et d’exemples de sorties.

---
'''
## Structure du projet


````

├── checkpoints/                   # logs de training & checkpoints
│   ├── latest.pt                 # dernier modèle entraîné (voir Drive)
│   └── \*.log                     # fichiers de logs (loss, FID, …)
├── configs/                      # fichiers de config YAML (dataset HES, modèle, etc.)
│   └── hes.yml
├── datasets/                     # dataloader et prétraitements HES
├── denoising\_diffusion\_pytorch/  # code source DDPM/DDIM (utils, training, sampling)
├── sample\_256/                   # sorties d’une génération d’image (256×256)
├── count\_params\_model.py         # script pour afficher le nombre de paramètres
├── evaluate\_hes.py               # calcul PSNR, SSIM, pieAPP, …
├── generate\_single\_DDPM\_fp32.py  # génération avec DDPM
├── generate\_single\_DDIM\_fp32.py  # génération avec DDIM
├── plot\_loss.py                  # tracé de la courbe de perte à partir des logs
├── plot\_fid.py                   # tracé de la courbe FID à partir des logs
├── train\_hes.py                  # script principal d’entraînement
└── setup.py                      # installation des dépendances

````

---

## Installation & Setup

1. **Cloner le dépôt**  
   ```bash
   git clone https://github.com/votre-utilisateur/votre-repo.git
   cd votre-repo
````

2. **Créer un environnement Python**
````
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Installer les dépendances**
   Toutes les dépendances sont listées dans `setup.py` :

   ```bash
   pip install -e .
   ```
4. **(Optionnel) Télécharger le checkpoint pré-entraîné**
   Le dernier modèle (`latest.pt`) est accessible ici :

   ```
   https://drive.google.com/drive/u/2/folders/1_IA5Se-CPvuphC82AojaTJ3QhQz688PS
   ```

   Placez ce fichier dans `checkpoints/latest.pt`.

---

## Entraînement

Lancez l’entraînement sur votre dataset HES avec la configuration par défaut :

```bash
python train_hes.py \
  --config configs/hes.yml \
  --batch_size 16 \
  --epochs 100 \
  --save_dir checkpoints
```

* Les checkpoints et logs (loss, FID) seront enregistrés dans `checkpoints/`.
* Un snapshot du modèle est sauvegardé toutes les 5 epochs (voir `checkpoints/`).

---

## Échantillonnage

### 1. Avec DDPM

```bash
python generate_single_DDPM_fp32.py \
  --config configs/hes.yml \
  --checkpoint checkpoints/latest.pt \
  --batch_size 1 \
  --output_dir sample_256 \
  --timesteps 1000 \
  --sampling_timesteps 1000 \
  --eta 0.0 \
  --seed 42
```

### 2. Avec DDIM

```bash
python generate_single_DDIM_fp32.py \
  --config configs/hes.yml \
  --checkpoint checkpoints/latest.pt \
  --batch_size 1 \
  --output_dir sample_256 \
  --timesteps 1000 \
  --seed 400
```

Les images générées seront déposées dans le dossier `sample_256/`.

---

## Évaluation & Visualisation des métriques

* **Loss & FID**

  ```bash
  python plot_loss.py --log_dir checkpoints
  python plot_fid.py  --log_dir checkpoints
  ```

  Ces scripts lisent les fichiers de logs dans `checkpoints/` et génèrent les courbes.

* **PSNR, SSIM, pieAPP, …**

  ```bash
  python evaluate_hes.py \
    --config configs/hes.yml \
    --checkpoint checkpoints/latest.pt \
    --data_dir path/vers/vos/images
  ```

  Affiche les métriques d’évaluation sur un ensemble d’images HES de référence.

---

## Autres scripts

* **`count_params_model.py`**
  Affiche le nombre total de paramètres du modèle DDPM.
* **`evaluate_hes.py`**
  Calcule PSNR, SSIM, pieAPP, etc., sur un dataset HES donné.
* **`plot_loss.py` & `plot_fid.py`**
  Génèrent respectivement les courbes de perte et de FID à partir des logs.

---

## Ressources & Lecture complémentaire

* [Article DDPM original (Ho et al.)](https://arxiv.org/abs/2006.11239)
* [Documentation PyTorch](https://pytorch.org/docs/stable/index.html)

---


