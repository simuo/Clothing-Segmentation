# import scipy.io as scio
# import random
# import numpy as np

# colors = []
# r = np.arange(0, 256, 50, dtype=np.int)
# g = np.arange(0, 256, 50, dtype=np.int)
# b = np.arange(0, 256, 50, dtype=np.int)
# for i in range(len(r) - 1):
#     for j in range(len(g) - 1):
#         for k in range(len(b) - 1):
#             colors.append([r[i], g[j], b[k]])
# random.shuffle(colors)
# print(colors)
# dataFile = '../label_list.mat'
# data = scio.loadmat(dataFile)
# cls = data['label_list'][0, :]
# clscolor = {}
# for i, name in enumerate(cls):
#     print(i)
#     # clscolor[f'{name[0]}']=colors[i]
#     clscolor[f'{i}'] = {}
#     clscolor[f'{i}'][f'{name[0]}'] = colors[i]
# with open('cfg.py', 'w') as f:
#     f.write(f'colors={clscolor}')
#     f.flush()



import scipy.io as scio
import random
import numpy as np

colors = []
r = np.arange(0, 256, 50, dtype=np.int)
g = np.arange(0, 256, 50, dtype=np.int)
b = np.arange(0, 256, 50, dtype=np.int)
for i in range(len(r) - 1):
    for j in range(len(g) - 1):
        for k in range(len(b) - 1):
            colors.append([r[i], g[j], b[k]])
random.shuffle(colors)
print(colors)
dataFile = '../label_list.mat'
data = scio.loadmat(dataFile)
cls = data['label_list'][0, :]
clscolor = {}
for i, name in enumerate(cls):
    print(i)
    # clscolor[f'{name[0]}']=colors[i]
    # clscolor[f'{i}'] = {}
    clscolor[f'{i}'] = colors[i]
with open('cfgnumber.py', 'w') as f:
    f.write(f'colors={clscolor}')
    f.flush()
