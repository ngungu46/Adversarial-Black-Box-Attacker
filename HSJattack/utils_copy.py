import tensorflow as tf
import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import math

import time as time

def load_image(image_path, image_size):
    raw_image = Image.open(image_path)

    transform = transforms.PILToTensor()
    image = transform(raw_image)

    image = image.to(dtype=torch.float32)
    image = image / 255
    transform = transforms.Resize(size=(image_size, image_size))
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)

    label = os.path.basename(os.path.dirname(image_path))

    return image, label


def display_images(model, image, save_path = None, image_name = None, suffix = ''):
    if suffix != '':
        suffix = f'_{suffix}'

    guess_data = model.predict(image)
    arr = []
    for guess in guess_data:
        print(guess[1] + ": " + str(guess[2]))
        arr.append(guess[1] + ": " + str(guess[2]))
    np.savetxt(f"{save_path}/top_k{suffix}.txt", np.array(arr), fmt='%s')
        
    plt.figure()
    plt.imshow(torch.permute(image[0], (1,2,0)))
 # plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                 #  label, confidence*100))
    # plt.show()
    if save_path is not None:
        plt.savefig(f"{save_path}/{image_name}{suffix}.jpg")

def norm(image, image2):
    y = image[0]
    z = image2[0]
    l2norm = torch.norm(np.subtract(z,y), p=2).numpy()
    infnorm = torch.norm(np.subtract(z,y), p=np.inf).numpy()
    return l2norm, infnorm

def select_delta(dist, d):
    # if cur_iter == 1:
    #     delta = 0.1
    # else:
        # delta = np.sqrt(d) * theta * dist
    delta = dist / d
    return delta

def clip_image(image, clip_min, clip_max):
	# Clip an image, or an image batch, with upper and lower threshold.
	return np.minimum(np.maximum(clip_min, image), clip_max)

def project(original_image, perturbed_image, alphas, dist, l):
    
    #alphas_shape = len(original_image.shape)
    #alphas = alphas.reshape(alphas_shape)
    if l == 'l2':
        return (1-alphas) * original_image + alphas * perturbed_image
    elif l == 'linf':
        out_images = clip_image(
            perturbed_image, 
            original_image - alphas * dist, 
            original_image + alphas * dist
        )
        return out_images
    
# def approximate_gradient(model, sample, num_evals, delta, l):
#     clip_max=1
#     clip_min =1

#     noise_shape = [num_evals] + list(sample.shape)[1:]
#     # print("noise shape", noise_shape)
#     if l == 'l2':
#         rv = np.random.randn(*noise_shape)
#     elif l == 'linf':
#         rv = np.random.uniform(low = -1, high = 1, size = noise_shape)
    
#     rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
#     perturbed = sample + delta * rv
#     # rv = (perturbed - sample) / delta

#     check2 = model.predict(sample)
#     check2 = np.array(check2[0][0])

#     for i in range(math.ceil(perturbed.shape[0]/256)):
#         if i == 0:
#             check1 = np.array([prob[0][0] for prob in model.predict(perturbed[i*256:i*256+256])])
#             decisions = (check1 != check2)
#         else:
#             check1 = np.array([prob[0][0] for prob in model.predict(perturbed[i*256:i*256+256])])
#             decisions = np.append(decisions, (check1 != check2), axis=0)
        
#     # check1 = np.array([prob[0][0] for prob in check1])
    
#     # decisions = (check1 != check2)

#     decision_shape = [len(decisions)] + [1] * (len(sample.shape) - 1)
#     fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

#     if np.mean(fval) == 1.0: # label changes. 
#         gradf = np.mean(rv, axis = 0)
#     elif np.mean(fval) == -1.0: # label not change.
#         gradf = - np.mean(rv, axis = 0)
#     else:
#         fval -= np.mean(fval)
#         gradf = np.mean(fval * rv, axis = 0) 

#     # Get the gradient direction.
#     gradf = gradf / np.linalg.norm(gradf)

#     return gradf

def approximate_gradient(model, sample, num_evals, delta, l):
    clip_max=1
    clip_min =1

    noise_shape = [num_evals] + list(sample.shape)[1:]
    # print("noise shape", noise_shape)
    if l == 'l2':
        rv = np.random.randn(*noise_shape)
    elif l == 'linf':
        rv = np.random.uniform(low = -1, high = 1, size = noise_shape)
    
    rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
    perturbed = sample + delta * rv

    decisions = []
    for i in range(math.ceil(perturbed.shape[0]/256)):
        out = np.array([1 if prob else 0 for prob in model.decision(perturbed[i*256:i*256+256])])
        decisions.append(out)
    decisions = np.concatenate(decisions, axis=0)

    decision_shape = [len(decisions)] + [1] * (len(sample.shape) - 1)
    fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

    if np.mean(fval) == 1.0: # label changes. 
        gradf = np.mean(rv, axis = 0)
    elif np.mean(fval) == -1.0: # label not change.
        gradf = - np.mean(rv, axis = 0)
    else:
        fval -= np.mean(fval)
        gradf = np.mean(fval * rv, axis = 0) 

    # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)

    return gradf

def geometric_progression(model, x, update, dist, cur_iter):
    epsilon = dist / np.sqrt(cur_iter) 

    def phi(epsilon):
        new = x + epsilon * update
        success = model.predict(new)
        # check1 = model.predict(new)
        # check2 = model.predict(x)
        # success = [check1[0][0] != check2[0][0]]
        return success

    while not phi(epsilon):
        epsilon /= 2.0

    return epsilon

def random_noise_hsja(model, image):
    image_size = image.shape[2]

    tries = 0
    while tries < 1000:
        tries += 1
        noise = np.random.uniform(0, 1, [1, 3, image_size, image_size])
        if model.decision(noise):
            break
    
    lo = 0.0
    hi = 1.0
    while hi - lo > 0.001:
        mid = (hi + lo) / 2.0
        blended = (1 - mid) * image + mid * noise 

        if model.decision(blended):
            hi = mid
        else:
            lo = mid
    
    final = (1 - hi) * image + hi * noise
    return final
    
# def binary_search_hsja(model, image, perturbed, theta, l='l2'):
#     distances = []
#     for p in perturbed:
#         if l == 'l2':
#             distances.append(norm(p, image)[0])
#         else:
#             distances.append(norm(p, image)[1])    
#     distances = np.array(distances)

#     if l == 'linf':
#         highs = distances
#         thresholds = np.minimum(distances * theta, theta)
#     else:
#         highs = np.ones(len(perturbed))
#         thresholds = theta

#     lows = np.zeros(len(perturbed))
    
#     while np.max((highs - lows) / thresholds) > 1:
        
#         mids = (highs + lows) / 2.0
        
#         decisions = np.array([])
        
#         for p in range(len(perturbed)):
#             mid_image = project(image, perturbed[p], mids[p], l)
#             d = model.decision(mid_image)
#             decisions = np.append(decisions, [d])
            
#         # Update highs and lows based on model decisions.
        
#         lows = np.where(decisions == 0, mids, lows)
#         highs = np.where(decisions == 1, mids, highs)


#     outputs = [project(image, perturbed[p], highs[p], l) for p in range(len(perturbed))]
    
#     finaldists = []
#     for p in perturbed:
#         if l == 'l2':
#             finaldists.append(norm(p, image)[0])
#         else:
#             finaldists.append(norm(p, image)[1])
            
#     idx = np.argmin(finaldists)

#     dist = distances[idx]
#     out_image = outputs[idx]
#     return out_image, dist

def binary_search_hsja(model, image, perturbed, theta, l='l2'):
    highs = np.ones(len(perturbed))
    lows = np.zeros(len(perturbed))

    if l == 'l2':
        distances = np.ones(len(perturbed))
    else: 
        distances = []
        for p in perturbed:
            distances.append(norm(p, image)[1])    
        distances = np.array(distances)
    
    while np.max((highs - lows)) > theta:
        mids = (highs + lows) / 2.0
        
        decisions = np.array([])
        
        for p in range(len(perturbed)):
            mid_image = project(image, perturbed[p], mids[p], distances[p], l)
            d = model.decision(mid_image)
            decisions = np.append(decisions, [d])
            
        # Update highs and lows based on model decisions.
        
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)

    outputs = [project(image, perturbed[p], highs[p], distances[p], l) for p in range(len(perturbed))]

    return outputs

def hsja(
    model, #instance of class Model
    image,
    constraint = 'l2',
    num_iterations = 30,
    gamma = 1,
    max_num_evals = 1e4,
    init_num_evals = 100,
    verbose = True,
    verbose_timing = False,
):
    d = np.prod(image.shape)
    
    if constraint == 'l2':
        theta = gamma / d**(3/2)
    else:
        theta = gamma / d**2
        
    perturbed = random_noise_hsja(model, image)

    if constraint == 'l2': 
        dist = norm(perturbed, image)[0]
    else:
        dist = norm(perturbed, image)[1]
    
    perturbed = binary_search_hsja(model, image, [perturbed], theta, constraint)[0]
    if constraint == 'l2': 
        dist_post = norm(perturbed, image)[0]
    else:
        dist_post = norm(perturbed, image)[1]
    print("Dist:", dist_post)
    
    for j in np.arange(num_iterations):
        start = time.time()
        c_iter = j + 1

        # Choose delta.
        start_time = time.time()
        delta = select_delta(dist_post, d)
        if verbose_timing:
            print('Select Delta Time:', time.time() - start_time)

        print("Delta:", delta)

        # Choose number of evaluations.
        num_evals = int(init_num_evals * np.sqrt(c_iter))
        num_evals = int(min([num_evals, max_num_evals]))
        
        # approximate gradient.
        start_time = time.time()
        gradf = approximate_gradient(model, perturbed, num_evals, 
            delta, constraint)
        if verbose_timing:
            print('Approximate Gradient Time:', time.time() - start_time)
        
        if constraint == 'linf':
            update = np.sign(gradf)
        else:
            update = gradf

        # find step size.
        start_time = time.time()

        epsilon = geometric_progression(model, perturbed, update, dist, c_iter)
        if verbose_timing:
            print('Geometric Progression Time:', time.time() - start_time)

        print("Epsilon:", epsilon)

        # Update the sample. 
        perturbed = perturbed + epsilon * update

        # Binary search to return to the boundary. 
        start_time = time.time()
        perturbed = binary_search_hsja(model, image, perturbed, theta, constraint)[0]
        if verbose_timing:
            print('Binary Search Time:', time.time() - start_time)
        
        # compute new distance.
        if constraint == 'l2': 
            dist_post = norm(perturbed, image)[0]
        else:
            dist_post = norm(perturbed, image)[1]
        print("Dist:", dist_post)
            
        if verbose:
            print(f'iteration: {j+1}, {constraint}')

        if verbose_timing:
            print(f"Iteration {j+1} time: {time.time()-start}")

        if constraint == 'l2':  
            dist = norm(perturbed, image)[0]
        else:
            dist = norm(perturbed, image)[1]
        
        print()

    return perturbed, dist

# def hsja(
#     model, #instance of class Model
#     image,
#     constraint = 'l2',
#     num_iterations = 30,
#     gamma = 1,
#     max_num_evals = 1e4,
#     init_num_evals = 100,
#     verbose = True,
#     verbose_timing = False,
# ):
#     d = np.prod(image.shape)
    
#     if constraint == 'l2':
#         theta = gamma / d**(3/2)
#     else:
#         theta = gamma / d**2
        
#     perturbed = random_noise_hsja(model, image)

#     if constraint == 'l2': 
#         dist = norm(perturbed, image)[0]
#     else:
#         dist = norm(perturbed, image)[1]
#     print("Dist:", dist)
    
#     perturbed = binary_search_hsja(model, image, [perturbed], theta, constraint)[0]
#     if constraint == 'l2': 
#         dist_post = norm(perturbed, image)[0]
#     else:
#         dist_post = norm(perturbed, image)[1]
    
#     for j in np.arange(num_iterations):
#         start = time.time()
#         c_iter = j + 1

#         # Choose delta.
#         start_time = time.time()
#         delta = select_delta(dist, d)
#         if verbose_timing:
#             print('Select Delta Time:', time.time() - start_time)

#         print("Delta:", delta)

#         # Choose number of evaluations.
#         num_evals = int(init_num_evals * np.sqrt(c_iter))
#         num_evals = int(min([num_evals, max_num_evals]))
        
#         # approximate gradient.
#         start_time = time.time()
#         gradf = approximate_gradient(model, perturbed, num_evals, 
#             delta, constraint)
#         if verbose_timing:
#             print('Approximate Gradient Time:', time.time() - start_time)
        
#         if constraint == 'linf':
#             update = np.sign(gradf)
#         else:
#             update = gradf

#         # find step size.
#         start_time = time.time()

#         if constraint == 'l2':  
#             dist = norm(perturbed, image)[0]
#         else:
#             dist = norm(perturbed, image)[1]

#         epsilon = geometric_progression(model, perturbed, update, dist, c_iter)
#         if verbose_timing:
#             print('Geometric Progression Time:', time.time() - start_time)

#         print("Epsilon:", epsilon)

#         # Update the sample. 
#         perturbed = perturbed + epsilon * update

#         # Binary search to return to the boundary. 
#         start_time = time.time()
#         perturbed = binary_search_hsja(model, image, perturbed, theta, constraint)[0]
#         if verbose_timing:
#             print('Binary Search Time:', time.time() - start_time)
        
#         # compute new distance.
#         if constraint == 'l2': 
#             dist = norm(perturbed, image)[0]
#         else:
#             dist = norm(perturbed, image)[1]
#         print("Dist:", dist)
            
#         if verbose:
#             print(f'iteration: {j+1}, {constraint}')

#         if verbose_timing:
#             print(f"Iteration {j+1} time: {time.time()-start}")
        
#         print()

#     return perturbed, dist
