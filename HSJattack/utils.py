import cv2
import random
import tensorflow as tf
import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import math

import time as time

def load_image(image_path, image_size):
    raw_image = Image.open(image_path)
    image = tf.keras.preprocessing.image.img_to_array(raw_image)
    
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.resize(image, (image_size, image_size))
    
    # Add batch dimension
    # 1, 299, 299, 3
    image = image[None, ...]

    if np.shape(image) == (1, image_size, image_size, 1):
        image = tf.image.grayscale_to_rgb(image)
    elif np.shape(image) == (1, image_size, image_size, 4):
        image = tf.image.grayscale_to_rgb(image)

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
    plt.imshow(image[0])
 # plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                 #  label, confidence*100))
    # plt.show()
    if save_path is not None:
        plt.savefig(f"{save_path}/{image_name}{suffix}.jpg")

def norm(image, image2):
    y = image[0]
    z = image2[0]
    l2norm = tf.norm(np.subtract(z,y), ord=2).numpy()
    infnorm = tf.norm(np.subtract(z,y), ord=np.inf).numpy()
    return l2norm, infnorm

def select_delta(dist, l, cur_iter, theta, d):
    if cur_iter == 1:
        delta = 0.1
    else:
        if l == 'l2':
            delta=np.sqrt(d)*theta*dist
        elif l == 'linf':
            delta=np.sqrt(d)*theta*dist
    return delta

def clip_image(image, clip_min, clip_max):
	# Clip an image, or an image batch, with upper and lower threshold.
	return np.minimum(np.maximum(clip_min, image), clip_max)

def project(original_image, perturbed_image, alphas, l):
    
    #alphas_shape = len(original_image.shape)
    #alphas = alphas.reshape(alphas_shape)
    if l == 'l2':
        return (1-alphas) * original_image + alphas * perturbed_image
    elif l == 'linf':
        out_images = clip_image(
            perturbed_image, 
            original_image - alphas, 
            original_image + alphas
        )
        return out_images
    
def approximate_gradient(model, sample, num_evals, delta, l):
    clip_max=1
    clip_min =1

    noise_shape = [num_evals] + list(sample.shape)
    # print("noise shape", noise_shape)
    if l == 'l2':
        rv = np.random.randn(*noise_shape)
    elif l == 'linf':
        rv = np.random.uniform(low = -1, high = 1, size = noise_shape)
    
    rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
    perturbed = sample + delta * rv
    rv = (perturbed - sample) / delta

    check2 = model.predict(sample)
    check2 = np.array(check2[0][0])

    print("Perturbed shape", perturbed.shape)
    print(perturbed[0*256:0*256+256, :].shape)
    for i in range(math.ceil(perturbed.shape[0]/256)):
        if i == 0:
            check1 = np.array([prob[0][0] for prob in model.predict(perturbed[i*256:i*256+256, :])])
            decisions = (check1 != check2)
        else:
            check1 = np.array([prob[0][0] for prob in model.predict(perturbed[i*256:i*256+256, :])])
            decisions = np.append(decisions, (check1 != check2), axis=0)
        
    # check1 = np.array([prob[0][0] for prob in check1])
    
    # decisions = (check1 != check2)

    decision_shape = [len(decisions)] + [1] * len(sample.shape)
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
        check1 = model.predict(new)
        check2 = model.predict(x)
        success = [check1[0][0] != check2[0][0]]
        return success

    while not phi(epsilon):
        epsilon /= 2.0

    return epsilon

def random_noise_hsja(model, image):
    image_size = image.shape[1]

    tries = 0
    while tries < 1000:
        tries += 1
        noise = np.random.uniform(0, 1, [1, image_size, image_size, 3])
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
    
def binary_search_hsja(model, image, perturbed, theta, l='l2'):
    distances = []
    for p in perturbed:
        if l == 'l2':
            distances.append(norm(p, image)[0])
        else:
            distances.append(norm(p, image)[1])    
    distances = np.array(distances)
    if l == 'linf':
        highs = distances
        thresholds = np.minimum(distances * theta, theta)
    else:
        highs = np.ones(len(perturbed))
        thresholds = theta

    lows = np.zeros(len(perturbed))
    
    while np.max((highs - lows) / thresholds) > 1:
        
        mids = (highs + lows) / 2.0
        
        decisions = np.array([])
        
        for p in range(len(perturbed)):
            mid_image = project(image, perturbed[p], mids[p], l)
            d = model.decision(mid_image)
            decisions = np.append(decisions, [d])
            
        # Update highs and lows based on model decisions.
        
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)


    outputs = [project(image, perturbed[p], highs[p], l) for p in range(len(perturbed))]
    
    finaldists = []
    for p in perturbed:
        if l == 'l2':
            finaldists.append(norm(p, image)[0])
        else:
            finaldists.append(norm(p, image)[1])
            
    idx = np.argmin(finaldists)

    dist = distances[idx]
    out_image = outputs[idx]
    return out_image, dist

def hsja(
    model, #instance of class Model
    image,
    constraint = 'l2',
    num_iterations = 30,
    gamma = 1,
    max_num_evals = 1e4,
    init_num_evals = 100,
    verbose = True
):
    d = np.prod(image.shape)
    
    if constraint == 'l2':
        theta = gamma / d**(3/2)
    else:
        theta = gamma / d**2
        
    perturbed = random_noise_hsja(model, image)
    
    perturbed, dist_post = binary_search_hsja(model, image, [perturbed], theta, constraint)
    
    if constraint == 'l2': 
        
        dist = norm(perturbed, image)[0]
        print(dist)
    else:
        dist = norm(perturbed, image)[1]
    
    for j in np.arange(num_iterations):
        start = time.time()
        print('start')
        c_iter = j + 1

        # Choose delta.
        start_time = time.time()
        delta = select_delta(dist, constraint, c_iter, theta, d)
        print('Select Delta Time:', time.time() - start_time)

        print("Delta" + str(delta))
        # Choose number of evaluations.
        num_evals = int(init_num_evals * np.sqrt(c_iter))
        num_evals = int(min([num_evals, max_num_evals]))
        
        # approximate gradient.
        start_time = time.time()
        gradf = approximate_gradient(model, perturbed, num_evals, 
            delta, constraint)
        print('Approximate Gradient Time:', time.time() - start_time)
        
        if constraint == 'linf':
            update = np.sign(gradf)
        else:
            update = gradf
        # find step size.
        start_time = time.time()
        epsilon = geometric_progression(model, perturbed, 
        update, dist, c_iter)
        print('Geometric Progression Time:', time.time() - start_time)
        print(epsilon)
        # Update the sample. 
        perturbed =  perturbed + epsilon * update

        # Binary search to return to the boundary. 
        start_time = time.time()
        perturbed, dist_post = binary_search_hsja(model, image, perturbed, theta, constraint)
        print('Binary Search Time:', time.time() - start_time)
        
        # compute new distance.
        if constraint == 'l2': 
            dist = norm(perturbed, image)[0]
        else:
            dist = norm(perturbed, image)[1]
            
        if verbose:
            print('iteration: {:d}, {:s} distance {:.4E}'.format(j+1, constraint, dist))

        print(f"Iteration {j+1} time: {time.time()-start}")

    return perturbed, dist