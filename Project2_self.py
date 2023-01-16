'''
Steps:
1. Get image
2. Find RG chromaticity
3. Get log chromaticity
4. Project on axis form 0 to 180 degrees
5. Make a histogram
6. Find minimum entropy
7. Find invariant image

TODO: geometric mean code, mean code, make plots
'''

from __future__ import division
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


U = np.array([ [1/math.sqrt(2), -1/math.sqrt(2), 0], [1/math.sqrt(6), 1/math.sqrt(6), -2/math.sqrt(6)] ]) # orthogonal matrix

def projectOntoPlane(log_points, orthMatrix):
    return log_points @ orthMatrix.T

# Step 5: Make histogram
# Step 6: Find minimum entropy
def ShannonEntropy(I, bandwidth = 1):
    nbins = round((np.max(I) - np.min(I)) / bandwidth)
    P = np.histogram(I, nbins)[0] / I.size
    P = P[P != 0]
    return -np.sum(P * np.log2(P))

# Step 2: Find 2D chromaticity values
    # Case A: Dividing by a single colour channel
    # Here c represents a string with possible values: R,G,B
def chrom_case_a(img,c):
    # splitting image into blue, green and red components
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    # changing value to 1.0 if pixel value is 0
    blue[blue == 0] = 1.0
    green[green == 0] = 1.0
    red[red == 0] = 1.0

    if (c=='R'):
        divisor = red
    elif(c=='B'):
        divisor = blue
    else:
        divisor = green

    chrom = np.zeros_like(img,dtype=np.float64)

    chrom[:,:,0] = blue/divisor
    chrom[:,:,1] = green/divisor
    chrom[:,:,2] = red/divisor

    return chrom

    # Case B: Dividing by mean of colour channels
def chrom_case_b(img):
    # splitting image into blue, green and red components
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    # changing value to 1.0 if pixel value is 0
    blue[blue == 0] = 1.0
    green[green == 0] = 1.0
    red[red == 0] = 1.0

    divisor = red+blue+green+1
    divisor[divisor==0] = 1.0

    chrom = np.zeros_like(img,dtype=np.float)

    chrom[:,:,0] = blue/divisor
    chrom[:,:,1] = green/divisor
    chrom[:,:,2] = red/divisor

    return chrom

    # Case C: Dividing by  geometric mean of colour channels
def chrom_case_c(img):
    # splitting image into blue, green and red components
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]
    
    # changing value to 1.0 if pixel value is 0
    blue[blue == 0] = 1.0
    green[green == 0] = 1.0
    red[red == 0] = 1.0
    divisor = np.power((red * green * blue), 1.0/3)
    chrom = np.zeros_like(img,dtype=np.float64)

    chrom[:,:,0] = blue/divisor
    chrom[:,:,1] = green/divisor
    chrom[:,:,2] = red/divisor

    return chrom


# Step 3: Get log chromaticity values (same for all 3 cases)
def get_log_chrom(chrom):
    log_chrom = np.zeros_like(chrom, dtype = np.float64)
    log_chrom[:, :, 0] = np.log(chrom[:,:,0])
    log_chrom[:, :, 1] = np.log(chrom[:,:,1])
    log_chrom[:, :, 2] = np.log(chrom[:,:,2])
    # display the log chromaticity image
    cv2.imshow("Log_chrom",log_chrom)

    # convert into 3d points
    log_chrom_b = np.atleast_3d(log_chrom[:, :, 0])
    log_chrom_g = np.atleast_3d(log_chrom[:, :, 1])
    log_chrom_r = np.atleast_3d(log_chrom[:, :, 2])
    print(log_chrom_b)
    # join the log values to get 3d points representing log_b,log_g, log_r values
    log_points = np.concatenate((log_chrom_r,log_chrom_g,log_chrom_b),axis=2)
    return log_chrom, log_points

# Step 4: Find point projections for all angles
# Step 5,6: Find min entropy and corresponding invariant angle
    # Case A: Dividing by a single colour channel
    # Here c represents a string with possible values: R,G,B
def projections_case_a(img,log_chrom,c):
    
    img_shape = img.shape[0]*img.shape[1]

    # initialising a list of entropies for angles from 0-180
    entropy_values = np.zeros(181, dtype = np.float64)

    # converting the angles to radians
    radians = np.radians(np.linspace(0,0,180))

    for i,theta in enumerate(radians):
        # I represents the projection of the 2D chromaticity point on a line with angle theta
        if (c=='R'):
            I = log_chrom[:,:,0]*np.cos(theta)+log_chrom[:,:,1]*np.sin(theta)     # (log b/r and log g/r)
        elif (c=='B'):
            I = log_chrom[:,:,1]*np.cos(theta)+log_chrom[:,:,2]*np.sin(theta)     # (log g/b and log r/b)
        else:
            I = log_chrom[:,:,0]*np.cos(theta)+log_chrom[:,:,2]*np.sin(theta)     # (log b/g and log r/g)

        # The following steps help clip the pixel values
        # between 5-95% to adjust for noise
        Mean = np.mean(I)
        Std_dev = np.std(I)
        lower_bound = Mean-(3*Std_dev)
        upper_bound = Mean+(3*Std_dev)
        clipped_values = np.clip(I,lower_bound,upper_bound)

        # calculating histogram bin size using the formula presented in paper
        bin_size = 3.5*Std_dev*img_shape**(-1/3)

        # calculating entropy value for the angle
        entropy_values[i] = ShannonEntropy(clipped_values,bin_size)

    # Find minimum entropy value and corresponding angle which will be the invariant angle
    min_entropy = np.min(entropy_values)
    min_ent_index = np.argmin(entropy_values)
    min_entropy_angle = radians[min_ent_index]
    invariant_angle = min_entropy_angle+ np.pi/2
    print ("Minimum Entropy: ",min_entropy)
    print ("Invariant Angle: ", invariant_angle)
    return invariant_angle

    # Case B,C
def projections(img,log_points):
    img_shape = img.shape[0]*img.shape[1]
    projected_points = projectOntoPlane(log_points, U)
    # initialising a list of entropies for angles from 0-180
    entropy_values = np.zeros(181, dtype = np.float64)

    # converting the angles to radians
    radians = np.radians(np.linspace(0,180,181))

    for i,theta in enumerate(radians):
        I = projected_points[:,:,0]*np.cos(theta) + projected_points[:,:,1]*np.sin(theta)
        # The following steps help clip the pixel values
        # between 5-95% to adjust for noise
        Mean = np.mean(I)
        Std_dev = np.std(I)
        lower_bound = Mean-(3.0*Std_dev)
        upper_bound = Mean+(3.0*Std_dev)
        clipped_values = np.clip(I,lower_bound,upper_bound)

        # calculating histogram bin size using the formula presented in paper
        bin_size = 3.5*Std_dev*img_shape**(-1/3)

        # calculating entropy value for the angle
        entropy_values[i] = ShannonEntropy(clipped_values,bin_size)

    # Find minimum entropy value and corresponding angle which will be the invariant angle
    min_entropy = np.min(entropy_values)
    min_ent_index = np.argmin(entropy_values)
    min_entropy_angle = radians[min_ent_index]
    # min_entropy_angle = 2.5830872929516078
    invariant_angle = min_entropy_angle+ np.pi/2
    print ("Minimum Entropy: ",min_entropy)
    print ("Invariant Angle: ", np.rad2deg(min_entropy_angle))
    return invariant_angle

# Step 7: Find invariant image
    # Case A: Dividing by a single colour channel
    # Here c represents a string with possible values: R,G,B
def invariant_image_case_a(log_chrom,invariant_angle,c):
    if (c=='R'):
        I = log_chrom[:,:,0]*np.cos(invariant_angle)+log_chrom[:,:,1]*np.sin(invariant_angle)     # (log b/r and log g/r)
    elif (c=='B'):
        I = log_chrom[:,:,1]*np.cos(invariant_angle)+log_chrom[:,:,2]*np.sin(invariant_angle)     # (log g/b and log r/b)
    else:
        I = log_chrom[:,:,0]*np.cos(invariant_angle)+log_chrom[:,:,2]*np.sin(invariant_angle)     # (log b/g and log r/g)

    # Finding inverse of log values
    I = np.exp(I)
    cv2.imshow("Invariant image",I)
    return I

    # Case B,C
def invariant_image(log_points,invariant_angle):
    projected_points = projectOntoPlane(log_points,U)
    # e = np.array([-1 * math.sin(invariant_angle), math.cos(invariant_angle)])

    # mat = np.array([np.cos(invariant_angle), np.sin(invariant_angle)])

    # Ptheta = np.outer(mat, mat)
    # I = projected_points[:,:,:2] @ Ptheta
    # R = I @ e
    # cv2.imshow("Pata nahi kya hai ye",R)

    # # average of the two channels of I coz I donno how to imshow 2 channel image
    # K = 0.5 * (I[:,:, 0] + I[:,:, 1])

    # cv2.imshow("log Inv image",K)

    I = projected_points[:,:,0]*np.cos(invariant_angle) + projected_points[:,:,1]*np.sin(invariant_angle) 
    I = np.exp(I)
    # print(K)

    # cv2.imshow("I1D", cv2.normalize(K, 0, 255, cv2.NORM_MINMAX))

    cv2.imshow("Invariant image",I)
    return I




            
