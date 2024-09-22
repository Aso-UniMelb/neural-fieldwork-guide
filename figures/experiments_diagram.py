import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
array = np.zeros((50, 25))

# Randomly choose 40 positions in the array
num_items = 40
indices = np.random.choice(50 * 25, num_items, replace=False)
for index in indices:
    row = index // 25
    col = index % 25
    array[row, col] = np.random.randint(1, 100)

plt.figure(figsize=(12, 6))
#=================
plt.subplot(1, 4, 1)
cmap = ListedColormap(['white', 'orange', 'blue', 'blue'])
plt.imshow(array, cmap=cmap)
plt.ylabel('lemmas')
plt.xlabel('tag sets')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)
plt.title("Exp.1")
#=================
plt.subplot(1, 4, 2)
cmap = ListedColormap(['white', 'orange', 'red' , 'green'])
plt.imshow(array, cmap=cmap)
plt.xlabel('tag sets')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)
plt.title("Exp.2")
#=================
plt.subplot(1, 4, 3)
data = np.random.rand(50, 25)
plt.imshow(data, cmap= 'PiYG')
plt.xlabel('tag sets')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color('black') 
    spine.set_linewidth(1)
plt.title("Exp.3")
#=================
array2 = np.zeros((50, 25))
indices = np.random.choice(50 * 25, 20, replace=False)
for index in indices:
    row = index // 25
    col = index % 25
    array2[row, col] = np.random.randint(50, 100)
for i in range(2):
    for j in range(25):
      array2[i, j] = 40
plt.subplot(1, 4, 4)
cmap = ListedColormap(['white', 'orange', 'red' , 'green'])
plt.imshow(array2, cmap=cmap)
plt.xlabel('tag sets')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)
plt.title("Exp.4")

plt.savefig(f'figures/experiments_diagram.png', dpi=300, bbox_inches='tight')