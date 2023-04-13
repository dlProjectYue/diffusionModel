import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
import os
import argparse
import numpy as np
import csv

class1=['1','2','3','4','5','6','7','8']
# int_class1=list(map(np.int64,class1))

file_pattern='*.txt'
real_data=None
label_list1=[]
for i in range (len(class1)):
    label=class1[i]
    folder_path=os.path.join('all_jam_AD/',class1[i])
    files=[f for f in os.listdir(folder_path) if f.endswith('.txt')]
    # real_data=np.array([])
    for j in range(len(files)):
        label_list1.append(int(label))
        file_path=os.path.join(folder_path,files[j])
        file_data=np.loadtxt(file_path)
        file_data=np.around(file_data,3)
        file_data = torch.from_numpy(file_data)
        file_data=(file_data-file_data.min())/(file_data.max()-file_data.min())
        # file_data=(file_data-torch.max(file_data))/(torch.max(file_data)-torch.min(file_data))
        file_data=file_data/10
        file_data=file_data.reshape(1,800)
        if real_data is None:
            real_data=file_data
        else:
            real_data=np.vstack((real_data,file_data))
n,m=real_data.shape
real_data=torch.tensor(real_data)
#如果是每个1行800列，竖直堆叠，就是n行800列，n代表realdata的个数
label_array1=np.array(label_list1)

fake_data=None
label_list2=[]
# class2=['tensor([0])','tensor([1])','tensor([2])','tensor([3])','tensor([4])','tensor([5])','tensor([6])','tensor([7])']
class2=['1']
for p in range (len(class2)):
    label=class2[p]
    # folder_path=os.path.join('gen_outputs/AD_1/',class2[p],'txt')
    folder_path = os.path.join('diffusion_4_output_800', class2[p], 'txt')
    files=[f for f in os.listdir(folder_path) if f.endswith('.txt')]
    # real_data=np.array([])
    for q in range(len(files)):
        label_list2.append(label)
        file_path=os.path.join(folder_path,files[q])
        file_data=np.loadtxt(file_path)
        # file_data=(file_data-torch.max(file_data))/(torch.max(file_data)-torch.min(file_data))
        # file_data=file_data/10
        file_data=file_data.reshape(1,800)
        if fake_data is None:
            fake_data=file_data
        else:
            fake_data=np.vstack((fake_data,file_data))
n,m=fake_data.shape
fake_data=torch.tensor(fake_data)
label_array2=np.array(label_list2)


tsne=TSNE(n_components=2,init='pca',random_state=0)
all_data=np.concatenate([real_data,fake_data],axis=0)
all_data=tsne.fit_transform(all_data)

all_data[:, 0] = (all_data[:, 0] - all_data[:, 0].min()) / (all_data[:, 0].max() - all_data[:, 0].min())
all_data[:, 1] = (all_data[:, 1] - all_data[:, 1].min()) / (all_data[:, 1].max() - all_data[:, 1].min())
real_data=all_data[:real_data.shape[0]]
fake_data=all_data[real_data.shape[0]:]

plt.figure(figsize=(10,10))
color_list = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'pink', 'orange']
for i in range(8):
    left=len(real_data[label_array1==(i+1),0])
    right=len(real_data[label_array1==(i+1),1])
    plt.scatter(real_data[label_array1 == (i+1), 0], real_data[label_array1 == (i+1), 1], c=color_list[i], marker='.',
                label=f'Class {i}')
for i in range(len(class2)):
    left=len(fake_data[label_array2==class2[i],0])
    right=len(fake_data[label_array2==class2[i],1])
    plt.scatter(fake_data[label_array2 == class2[i], 0], fake_data[label_array2 == class2[i], 1], c=color_list[i], marker='x',
                label=f'Class {i}')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.rcParams.update({'font.size':15})
plt.tick_params(axis='both',width=1,length=5)
plt.legend()
plt.show()