
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
            # print(data)

        status = data['status']
        if status != 'FAILED':
            distance = data['L_inf_distance1']
            distances.append(distance)
        
        count += 1
    
    distances = [0.07545924575071213, 0.1093171326188532, 0.17757503932190122, 0.10089768745723368, 0.10065678497743237, 0.24346760936500228, 0.15036326233859407, 0.04879804446535825, 0.08764603887426299, 0.10085872613271776, 0.11890091945414771, 0.11257668727851237, 0.17002597840939174, 0.1409734422633661, 0.215193839825176, 0.11686851018297514, 0.046339145359233024, 0.04430218052816451, 0.04234084782302363, 0.046814857903300804, 0.06716754015208962, 0.1525849295634762, 0.06955852127016075, 0.09494961076865213, 0.24325968199096842, 0.04490040152644805, 0.04054776182768223, 0.09899622632351433, 0.10500976271136654, 0.11874901744946588, 0.10308576525711227, 0.12069327851498324, 0.06799582831102752, 0.05856664921913378, 0.09965062182651531, 0.15710318319162975, 0.08089177667652406, 0.06282964542286014, 0.05819408758084967, 0.05675612121300424, 0.14588724810195122, 0.12240865197854178, 0.09776706006503794, 0.05725370485961857, 0.12956229200327876, 0.06519518231659976, 0.09577020746669035, 0.11016182275550757, 0.15762339121889454, 0.09648458245338778, 0.15511854146094445, 0.05865226775977805, 0.09599682954208591, 0.10411942747486054, 0.1373300226277526, 0.04897982692338938, 0.11307418346802223, 0.1336139870292266, 0.1256225086521675, 0.07602047329495676, 0.12986791069230832, 0.24299146567776206, 0.07224767459437371, 0.055740046972664534, 0.08655606430448759, 0.15988790019033167, 0.051281883754812005, 0.09730063836553, 0.11960542961169955, 0.13687379982711168, 0.10483690440478574, 0.06823538868812347, 0.12262164506172991, 0.06573172584856045, 0.11779709158447595, 0.17735738248798166, 0.20143063761949398, 0.0929968942206688, 0.0545911097377565, 0.14536252956941512, 0.09396259296155512, 0.048127228103945645, 0.05270712406889039, 0.20272598526012486, 0.09919783550441406, 0.05462757070118268, 0.11085435820547612, 0.05827644594821773, 0.04868221774374998] 

    # print(len(distances))

    distances = np.array(distances)
    distances.sort()
    return distances, count

def plot(distances, count, label):
    length = distances.shape[0]

    distances = np.append(distances, [1])
    plt.plot(distances, np.arange(length+1) / count, label=label)

# distances, count = load_hsja('imagenet', False)
# plot(distances, count, 'No Defense')

# distances, count = load_hsja('imagenet', True)
# plot(distances, count, 'Sine Defense')

# plt.title('HSJA Imagenet')

# distances, count = load_hsja('butterfly', False)
# plot(distances, count, 'No Defense')

distances, count = load_hsja('butterfly', True)
plot(distances, count, 'Sine Defense')

plt.title('HSJA Butterfly')

# distances, count = load_nes()
# plot(distances, count, 'NES No Defense')

plt.xlabel(r'L$\infty$ Distance')
plt.ylabel('Success Rate')
plt.legend()

# plt.savefig(f'output/graph_hsja_butterfly')

print(distances.mean())