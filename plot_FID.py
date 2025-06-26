import os
import matplotlib.pyplot as plt

# Chemin vers le fichier de log
path = 'checkpoints/fid_log.txt'

# Lecture du fichier ou données d'exemple si absent
if os.path.exists(path):
    with open(path, 'r') as f:
        lines = f.readlines()
else:
    lines = [
        'Epoch: 5, iter: 7590, FID: 361.9117',
        'Epoch: 10, iter: 15180, FID: 254.1988',
        'Epoch: 15, iter: 22770, FID: 186.6587',
    ]

# Extraction des epochs et des FID
epochs = []
fids = []
for line in lines:
    parts = line.strip().split(',')
    epoch = int(parts[0].split(':')[1].strip())
    fid = float(parts[2].split(':')[1].strip())
    epochs.append(epoch)
    fids.append(fid)

# Tracé de la courbe
plt.figure()
plt.plot(epochs, fids)
plt.xlabel('Epoch')
plt.ylabel('FID')
plt.title('FID vs. Epoch')
plt.grid(True)
plt.show()
