# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:01:02 2015

@author: lc585

Estimating ranges with the zscale algorithm

@type input_arr: numpy array
@param input_arr: image data array as sample pixels to derive z-ranges
@type contrast: float
@param contrast: zscale contrast which should be larger than 0.
@type sig_fract: float
@param sig_fract: fraction of sigma clipping
@type percent_fract: float
@param percent_fract: convergence fraction
@type max_iter: integer
@param max_iter: max. of iterations
@type low_cut: boolean
@param low_cut: cut out only low values
@type high_cut: boolean
@param high_cut: cut out only high values
@rtype: tuple
@return: (min. value, max. value, number of iterations)

"""


from __future__ import division

import numpy as np
import math

def range_from_zscale(input_arr,
                      contrast = 1.0,
                      sig_fract = 3.0,
                      percent_fract = 0.01,
                      max_iter=100,
                      low_cut=True,
                      high_cut=True):


	work_arr = np.ravel(input_arr)
	work_arr = np.sort(work_arr) # sorting is done.
	max_ind = len(work_arr) - 1
	midpoint_ind = int(len(work_arr)*0.5)
	I_midpoint = work_arr[midpoint_ind]
	# print ".. midpoint index ", midpoint_ind, " I_midpoint ", I_midpoint
	# initial estimation of the slope
	x = np.array(range(0, len(work_arr))) - midpoint_ind
	y = np.array(work_arr)
	temp = np.vstack([x, np.ones(len(x))]).T
	slope, intercept = np.linalg.lstsq(temp, y)[0]
	old_slope = slope
	# print "... slope & intercept ", old_slope, " ", intercept
	# initial clipping
	sig = y.std()
	upper_limit = I_midpoint + sig_fract * sig
	lower_limit = I_midpoint - sig_fract * sig
	if low_cut and high_cut:
		indices = np.where((work_arr < upper_limit) & (work_arr > lower_limit))
	else:
		if low_cut:
			indices = np.where((work_arr > lower_limit))
		else:
			indices = np.where((work_arr < upper_limit))
	# new estimation of the slope
	x = np.array(indices[0]) - midpoint_ind
	y = np.array(work_arr[indices])
	temp = np.vstack([x, np.ones(len(x))]).T
	slope, intercept = np.linalg.lstsq(temp, y)[0]
	new_slope = slope
	# print "... slope & intercept ", new_slope, " ", intercept
	iteration = 1
	# to run the iteration, we need more than 50% of the original input array
	while (((math.fabs(old_slope - new_slope)/new_slope) > percent_fract) and (iteration < max_iter)) and (len(y) >= midpoint_ind) :
		iteration += 1
		old_slope = new_slope
		# clipping
		sig = y.std()
		upper_limit = I_midpoint + sig_fract * sig
		lower_limit = I_midpoint - sig_fract * sig
		if low_cut and high_cut:
			indices = np.where((work_arr < upper_limit) & (work_arr > lower_limit))
		else:
			if low_cut:
				indices = np.where((work_arr > lower_limit))
			else:
				indices = np.where((work_arr < upper_limit))
		# new estimation of the slope
		x = np.array(indices[0]) - midpoint_ind
		y = work_arr[indices]
		temp = np.vstack([x, np.ones(len(x))]).T
		slope, intercept = np.linalg.lstsq(temp, y)[0]
		new_slope = slope
		# print "... slope & intercept ", new_slope, " ", intercept

	z1 = I_midpoint + (new_slope / contrast) * (0 - midpoint_ind)
	z2 = I_midpoint + (new_slope / contrast) * (max_ind - midpoint_ind)

	return (z1, z2, iteration)
