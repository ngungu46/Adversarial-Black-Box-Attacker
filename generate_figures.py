
import numpy as np

import matplotlib.pyplot as plt

import json

import os


def load_hsja(dataset, has_defender):
    result_path = f'./output/hsja_linf_{dataset}_{"sine" if has_defender else "none"}'

    count = 0
    distances = []
    for image in os.listdir(result_path):
        distance = np.loadtxt(f'{result_path}/{image}/distance{"_defense" if has_defender else ""}.txt')
        distances.append(distance[-1])
        
        count += 1

    distances = np.array(distances)
    distances.sort()
    return distances, count

def load_nes():
    result_path = './NESattack/outputs/po_imagenet_0'

    count = 0
    distances = []
    for file in os.listdir(result_path):
        with open(f'{result_path}/{file}') as f:
            data = json.load(f)
            print(data)

        status = data['status']
        if status != 'FAILED':
            distance = data['L_inf_distance1']
            distances.append(distance)
        
        count += 1

    distances = np.array(distances)
    distances.sort()
    return distances, count

def plot(distances, count, label):
    length = distances.shape[0]

    distances = np.append(distances, [1])
    plt.plot(distances, np.arange(length+1) / count, label=label)

dataset = 'imagenet'
has_defender = False

# distances, count = load_hsja(dataset, False)
distances, count = load_nes()
plot(distances, count, 'HSJA No Defense')

# distances, count = load_hsja(dataset, True)
# plot(distances, count, 'HSJA Sine Defense')

# distances, count = load_hsja('butterfly', False)

plt.xlabel('Linf Distance')
plt.ylabel('Success Rate')
plt.legend()

plt.savefig(f'output/graph')

print(distances)
print(distances.mean())