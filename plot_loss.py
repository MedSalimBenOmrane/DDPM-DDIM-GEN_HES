import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Chemin complet vers votre fichier de log
log_file = 'checkpoints/loss_log.txt'

# Dictionnaire pour stocker les losses par epoch
losses = defaultdict(list)

# Lecture et parsing du fichier
with open(log_file, 'r') as f:
    for line in f:
        m = re.match(r"Epoch:\s*(\d+),\s*iter:\s*(\d+),\s*Loss:\s*([\d\.]+)", line)
        if m:
            epoch = int(m.group(1))
            loss = float(m.group(3))
            losses[epoch].append(loss)

# Calcul de la loss moyenne par epoch
epochs = sorted(losses.keys())
avg_losses = [np.mean(losses[e]) for e in epochs]

# Affichage du graphique avec points pour chaque epoch
plt.figure(figsize=(8,5))
plt.plot(epochs, avg_losses, '-o', linewidth=1, markersize=4)  # '-o' ajoute les markers
plt.xlabel('Epoch')
plt.ylabel('Loss Moyenne')
plt.title('Loss Moyenne par Epoch')
plt.grid(True)
plt.tight_layout()
plt.show()
