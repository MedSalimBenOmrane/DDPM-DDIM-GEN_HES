<<<<<<< HEAD
<img src="./images/denoising-diffusion.png" width="500px"></img>

## Denoising Diffusion Probabilistic Model, in Pytorch

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch. It is a new approach to generative modeling that may <a href="https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/">have the potential</a> to rival GANs. It uses denoising score matching to estimate the gradient of the data distribution, followed by Langevin sampling to sample from the true distribution.

This implementation was inspired by the official Tensorflow version <a href="https://github.com/hojonathanho/diffusion">here</a>

Youtube AI Educators - <a href="https://www.youtube.com/watch?v=W-O7AZNzbzQ">Yannic Kilcher</a> | <a href="https://www.youtube.com/watch?v=344w5h24-h8">AI Coffeebreak with Letitia</a> | <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">Outlier</a>

<a href="https://github.com/yiyixuxu/denoising-diffusion-flax">Flax implementation</a> from <a href="https://github.com/yiyixuxu">YiYi Xu</a>

<a href="https://huggingface.co/blog/annotated-diffusion">Annotated code</a> by Research Scientists / Engineers from <a href="https://huggingface.co/">🤗 Huggingface</a>

Update: Turns out none of the technicalities really matters at all | <a href="https://arxiv.org/abs/2208.09392">"Cold Diffusion" paper</a> | <a href="https://muse-model.github.io/">Muse</a>

<img src="./images/sample.png" width="500px"><img>

[![PyPI version](https://badge.fury.io/py/denoising-diffusion-pytorch.svg)](https://badge.fury.io/py/denoising-diffusion-pytorch)

## Install

```bash
$ pip install denoising_diffusion_pytorch
```

## Usage

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
```

Or, if you simply want to pass in a folder name and the desired image dimensions, you can use the `Trainer` class to easily train a model.

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()
```

Samples and model checkpoints will be logged to `./results` periodically

## Multi-GPU Training

The `Trainer` class is now equipped with <a href="https://huggingface.co/docs/accelerate/accelerator">🤗 Accelerator</a>. You can easily do multi-gpu training in two steps using their `accelerate` CLI

At the project root directory, where the training script is, run

```python
$ accelerate config
```

Then, in the same directory

```python
$ accelerate launch train.py
```

## Miscellaneous

### 1D Sequence

By popular request, a 1D Unet + Gaussian Diffusion implementation.

```python
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1

loss = diffusion(training_seq)
loss.backward()

# Or using trainer

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)

```

`Trainer1D` does not evaluate the generated samples in any way since the type of data is not known.

You could consider adding a suitable metric to the training loop yourself after doing an editable install of this package
`pip install -e .`.

## Citations

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

```bibtex
@InProceedings{pmlr-v139-nichol21a,
    title       = {Improved Denoising Diffusion Probabilistic Models},
    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
    pages       = {8162--8171},
    year        = {2021},
    editor      = {Meila, Marina and Zhang, Tong},
    volume      = {139},
    series      = {Proceedings of Machine Learning Research},
    month       = {18--24 Jul},
    publisher   = {PMLR},
    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
    url         = {https://proceedings.mlr.press/v139/nichol21a.html},
}
```

```bibtex
@inproceedings{kingma2021on,
    title       = {On Density Estimation with Diffusion Models},
    author      = {Diederik P Kingma and Tim Salimans and Ben Poole and Jonathan Ho},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year        = {2021},
    url         = {https://openreview.net/forum?id=2LdBqxc1Yv}
}
```

```bibtex
@article{Karras2022ElucidatingTD,
    title   = {Elucidating the Design Space of Diffusion-Based Generative Models},
    author  = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00364}
}
```

```bibtex
@article{Song2021DenoisingDI,
    title   = {Denoising Diffusion Implicit Models},
    author  = {Jiaming Song and Chenlin Meng and Stefano Ermon},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2010.02502}
}
```

```bibtex
@misc{chen2022analog,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey Hinton},
    year    = {2022},
    eprint  = {2208.04202},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Salimans2022ProgressiveDF,
    title   = {Progressive Distillation for Fast Sampling of Diffusion Models},
    author  = {Tim Salimans and Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.00512}
}
```

```bibtex
@article{Ho2022ClassifierFreeDG,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2207.12598}
}
```

```bibtex
@article{Sunkara2022NoMS,
    title   = {No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects},
    author  = {Raja Sunkara and Tie Luo},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.03641}
}
```

```bibtex
@inproceedings{Jabri2022ScalableAC,
    title   = {Scalable Adaptive Computation for Iterative Generation},
    author  = {A. Jabri and David J. Fleet and Ting Chen},
    year    = {2022}
}
```

```bibtex
@article{Cheng2022DPMSolverPlusPlus,
    title   = {DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models},
    author  = {Cheng Lu and Yuhao Zhou and Fan Bao and Jianfei Chen and Chongxuan Li and Jun Zhu},
    journal = {NeuRips 2022 Oral},
    year    = {2022},
    volume  = {abs/2211.01095}
}
```

```bibtex
@inproceedings{Hoogeboom2023simpleDE,
    title   = {simple diffusion: End-to-end diffusion for high resolution images},
    author  = {Emiel Hoogeboom and Jonathan Heek and Tim Salimans},
    year    = {2023}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.01327,
    doi     = {10.48550/ARXIV.2302.01327},
    url     = {https://arxiv.org/abs/2302.01327},
    author  = {Kumar, Manoj and Dehghani, Mostafa and Houlsby, Neil},
    title   = {Dual PatchNorm},
    publisher = {arXiv},
    year    = {2023},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

```bibtex
@inproceedings{Hang2023EfficientDT,
    title   = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
    author  = {Tiankai Hang and Shuyang Gu and Chen Li and Jianmin Bao and Dong Chen and Han Hu and Xin Geng and Baining Guo},
    year    = {2023}
}
```

```bibtex
@misc{Guttenberg2023,
    author  = {Nicholas Guttenberg},
    url     = {https://www.crosslabs.org/blog/diffusion-with-offset-noise}
}
```

```bibtex
@inproceedings{Lin2023CommonDN,
    title   = {Common Diffusion Noise Schedules and Sample Steps are Flawed},
    author  = {Shanchuan Lin and Bingchen Liu and Jiashi Li and Xiao Yang},
    year    = {2023}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Bondarenko2023QuantizableTR,
    title   = {Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing},
    author  = {Yelysei Bondarenko and Markus Nagel and Tijmen Blankevoort},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2306.12929},
    url     = {https://api.semanticscholar.org/CorpusID:259224568}
}
```

```bibtex
@article{Karras2023AnalyzingAI,
    title   = {Analyzing and Improving the Training Dynamics of Diffusion Models},
    author  = {Tero Karras and Miika Aittala and Jaakko Lehtinen and Janne Hellsten and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2312.02696},
    url     = {https://api.semanticscholar.org/CorpusID:265659032}
}
```

```bibtex
@article{Li2024ImmiscibleDA,
    title   = {Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment},
    author  = {Yiheng Li and Heyang Jiang and Akio Kodaira and Masayoshi Tomizuka and Kurt Keutzer and Chenfeng Xu},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.12303},
    url     = {https://api.semanticscholar.org/CorpusID:270562607}
}
```

```bibtex
@article{Chung2024CFGMC,
    title   = {CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models},
    author  = {Hyungjin Chung and Jeongsol Kim and Geon Yeong Park and Hyelin Nam and Jong Chul Ye},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2406.08070},
    url     = {https://api.semanticscholar.org/CorpusID:270391454}
}
```

```bibtex
@inproceedings{Sadat2024EliminatingOA,
    title   = {Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models},
    author  = {Seyedmorteza Sadat and Otmar Hilliges and Romann M. Weber},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273098845}
}
```
=======
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


>>>>>>> origin/main
