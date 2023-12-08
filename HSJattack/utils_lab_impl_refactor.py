
import numpy as np

def hsja(model, sample, constraint = 'l2', num_iterations = 40):
		
	# Set parameters
	original_label = model.label
	params = {	'original_label': original_label, 
				'constraint': constraint,
				'd': int(np.prod(sample.shape)), 
				}

	# Set binary search threshold.
	if params['constraint'] == 'l2':
		params['theta'] = 1.0 / (np.sqrt(params['d']) * params['d'])
	else:
		params['theta'] = 1.0 / (params['d'] ** 2)
		
	# Initialize.
	perturbed = initialize(model, sample)
	

	# Project the initialization to the boundary.
	perturbed, dist_post_update = binary_search_batch(sample, 
		np.expand_dims(perturbed, 0), 
		model, 
		params)
	dist = compute_distance(perturbed, sample, constraint)

	for j in np.arange(num_iterations):
		# pass
		params['cur_iter'] = j + 1

		# Choose delta.
		delta = select_delta(params, dist_post_update)

		# Choose number of evaluations.
		num_evals = int(100 * np.sqrt(j+1))
		num_evals = int(min([num_evals, 1e4]))

		# approximate gradient.
		gradf = approximate_gradient(model, perturbed, num_evals, delta, params)
		if params['constraint'] == 'linf':
			update = np.sign(gradf)
		else:
			update = gradf
		# search step size.
        # find step size.
        
		epsilon = geometric_progression_for_stepsize(perturbed, update, dist, model, params)

        # Update the sample. 
		perturbed = clip_image(perturbed + epsilon * update)

        # Binary search to return to the boundary. 
		perturbed, dist_post_update = binary_search_batch(sample, perturbed[None], model, params)

		# compute new distance.
		dist = compute_distance(perturbed, sample, constraint)
		
		print('iteration: {:d}, {:s} distance {:.4E}'.format(j+1, constraint, dist))

	return perturbed, dist

def decision_function(model, images):
	"""
	Decision function output 1 on the desired side of the boundary,
	0 otherwise.
	"""
	images = clip_image(images)

	decisions = model.decision(images)
	decisions = np.array(decisions)

	return decisions
	# prob = model.predict(images)
	# if params['target_label'] is None:
	# 	return np.argmax(prob, axis = 1) != params['original_label'] 
	# else:
	# 	return np.argmax(prob, axis = 1) == params['target_label']

def clip_image(image):
	# Clip (batch) images to [0, 1]
	return np.minimum(np.maximum(0, image), 1) 


def compute_distance(x_ori, x_pert, constraint = 'l2'):
	# Compute the distance between two images.
	if constraint == 'l2':
		return np.linalg.norm(x_ori - x_pert)
	elif constraint == 'linf':
		return np.max(abs(x_ori - x_pert))


def approximate_gradient(model, sample, num_evals, delta, params):
	# Generate gradient approximation
	noise_shape = [num_evals] + list(sample.shape)
	if params['constraint'] == 'l2':
		rv = np.random.randn(*noise_shape)
	elif params['constraint'] == 'linf':
		rv = np.random.uniform(low = -1, high = 1, size = noise_shape)

	rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
	perturbed = sample + delta * rv
	perturbed = clip_image(perturbed)
	rv = (perturbed - sample) / delta

	# query the model.
	decisions = decision_function(model, perturbed)
	decision_shape = [len(decisions)] + [1] * len(sample.shape)
	fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

	# Baseline subtraction (when fval differs)
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


def project(original_image, perturbed_images, alphas, params):
	alphas_shape = [len(alphas)] + [1] * len(perturbed_images.shape)
	alphas = alphas.reshape(alphas_shape)
	if params['constraint'] == 'l2':
		return (1-alphas) * original_image + alphas * perturbed_images
	elif params['constraint'] == 'linf':
		out_images = clip_image(
			perturbed_images, 
			original_image - alphas, 
			original_image + alphas
			)
		return out_images


def binary_search_batch(original_image, perturbed_images, model, params):
	""" Binary search to approach the boundar. """

	# Compute distance between each of perturbed image and original image.
	dists_post_update = np.array([
			compute_distance(
				original_image, 
				perturbed_image, 
				params['constraint']
			) 
			for perturbed_image in perturbed_images])

	# Choose upper thresholds in binary searchs based on constraint.
	if params['constraint'] == 'linf':
		highs = dists_post_update
		# Stopping criteria.
		thresholds = np.minimum(dists_post_update * params['theta'], params['theta'])
	else:
		highs = np.ones(len(perturbed_images))
		thresholds = params['theta']

	lows = np.zeros(len(perturbed_images))

	# Call recursive function. 
	while np.max((highs - lows) / thresholds) > 1:
		# projection to mids.
		mids = (highs + lows) / 2.0
		mid_images = project(original_image, perturbed_images, mids, params)

		# Update highs and lows based on model decisions.
		decisions = decision_function(model, mid_images)
		lows = np.where(not decisions, mids, lows)
		highs = np.where(decisions, mids, highs)

	out_images = project(original_image, perturbed_images, highs, params)

	# Compute distance of the output image to select the best choice. 
	# (only used when stepsize_search is grid_search.)
	dists = np.array([
		compute_distance(
			original_image, 
			out_image, 
			params['constraint']
		) 
		for out_image in out_images])
	idx = np.argmin(dists)

	dist = dists_post_update[idx]
	out_image = out_images[idx]
	return out_image, dist


def initialize(model, sample):
	""" 
	Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
	"""
	success = 0
	num_evals = 0

	# Find a misclassified random noise for untargeted attack.
	while True:
		random_noise = np.random.uniform(0, 
			1, size = sample.shape)
		success = decision_function(model,random_noise[None])[0]
		num_evals += 1
		if success:
			break
		assert num_evals < 1e4,"Initialization failed! "
		"Use a misclassified image as `target_image`" 


	# Binary search to minimize l2 distance to original image.
	low = 0.0
	high = 1.0
	while high - low > 0.001:
		mid = (high + low) / 2.0
		blended = (1 - mid) * sample + mid * random_noise 
		success = decision_function(model, blended[None])
		if success:
			high = mid
		else:
			low = mid

	initialization = (1 - high) * sample + high * random_noise 

	return initialization


def geometric_progression_for_stepsize(x, update, dist, model, params):
	"""
	Geometric progression to search for stepsize.
	Keep decreasing stepsize by half until reaching 
	the desired side of the boundary,
	"""
	epsilon = dist / np.sqrt(params['cur_iter']) 

	def phi(epsilon):
		new = x + epsilon * update
		success = decision_function(model, new[None])
		return success

	while not phi(epsilon):
		epsilon /= 2.0

	return epsilon

def select_delta(params, dist_post_update):
	""" 
	Choose the delta at the scale of distance 
	between x and perturbed sample. 

	"""
	if params['cur_iter'] == 1:
		delta = 0.1
	else:
		if params['constraint'] == 'l2':
			delta = np.sqrt(params['d']) * params['theta'] * dist_post_update
		elif params['constraint'] == 'linf':
			delta = params['d'] * params['theta'] * dist_post_update	

	return delta


