'''
Steps:
1. Get image
2. Find RG chromaticity
3. Get log chromaticity
4. Project on axis form 0 to 180 degrees
5. Make a histogram
6. Find minimum entropy
7. Find invariant image
'''


import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image

# Step 5: Make histogram
# Step 6: Find minimum entropy
def ShannonEntropy(I, bandwidth = 1):
    nbins = round((np.max(I) - np.min(I)) / bandwidth)
    P = np.histogram(I, nbins)[0] / I.size
    P = P[P != 0]
    return -np.sum(P * np.log2(P))

# Step 2: Find BG Chromaticity
def find_bg_chrom(img):

    # splitting image into blue, green and red components
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    # changing value to 1.0 if pixel value is 0
    B[B == 0] = 1.0
    G[G == 0] = 1.0
    R[R == 0] = 1.0

    divisor = R
    chrom = np.zeros_like(img,dtype=np.float64)

    chrom[:,:,0] = B/divisor
    chrom[:,:,1] = G/divisor
    chrom[:,:,2] = R/divisor

    return chrom

def find_gr_chrom(img):
    # splitting image into blue, green and red components
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    # changing value to 1.0 if pixel value is 0
    B[B == 0] = 1.0
    G[G == 0] = 1.0
    R[R == 0] = 1.0

    divisor = B
    chrom = np.zeros_like(img,dtype=np.float64)

    chrom[:,:,0] = B/divisor
    chrom[:,:,1] = G/divisor
    chrom[:,:,2] = R/divisor

    return chrom

def find_br_chrom(img):
    # splitting image into blue, green and red components
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    # changing value to 1.0 if pixel value is 0
    B[B == 0] = 1.0
    G[G == 0] = 1.0
    R[R == 0] = 1.0

    divisor = G
    chrom = np.zeros_like(img,dtype=np.float64)

    chrom[:,:,0] = B/divisor
    chrom[:,:,1] = G/divisor
    chrom[:,:,2] = R/divisor

    return chrom


# def bgr_mean_chrom(img):


# Step 3: Get log chromaticity values
def get_log_chrom(chrom):
    logChr = np.zeros_like(img, dtype = np.float64)
    logChr[:, :, 0] = np.log(chrom[:,:,0])
    logChr[:, :, 1] = np.log(chrom[:,:,1])
    logChr[:, :, 2] = np.log(chrom[:,:,2])

    cv2.imshow("Log_chrom",logChr)

    # # convert into 2d points, only for BG case
    # logChrB = np.atleast_3d(logChr[:, :, 0])
    # logChrG = np.atleast_3d(logChr[:, :, 1])
    # logChrR = np.atleast_3d(logChr[:, :, 2])

    # # join the log values to get 2d points representing log_b,log_g values [2d log plane]
    # log_points = np.concatenate((logChrB,logChrG,logChrR),axis=2)

    return logChr

# Step 4: Find point projections for all angles
# Step 5,6: Find min entropy and corresponding invariant angle
def projections(img,logChr):
    # projected_log_points = projectOnPlane(log_points,U)
    img_shape = img.shape[0]*img.shape[1]
    Entropies = np.zeros(181, dtype = np.float64)
    radians = np.radians(np.linspace(0,180,181))
    for i,rad in enumerate(radians):
        I = logChr[:,:,0]*np.cos(rad)+logChr[:,:,1]*np.sin(rad)
        Mean = np.mean(I)
        Std_dev = np.std(I)
        lower_bound = Mean-(3*Std_dev)
        upper_bound = Mean+(3*Std_dev)
        Clipped_pixels = np.clip(I,lower_bound,upper_bound)
        bin_size = 3.5*Std_dev*img_shape**(-1/3)
        Entropies[i] = ShannonEntropy(Clipped_pixels,bin_size)

    min_entropy = np.min(Entropies)
    min_ent_index = np.argmin(Entropies)
    min_entropy_angle = radians[min_ent_index]
    invariant_angle = min_entropy_angle+ np.pi/2
    print ("Minimum Entropy: ",min_entropy)
    print ("Invariant Angle: ", invariant_angle)
    return invariant_angle

def projected_image(logChr,invariant_angle):
    I = logChr[:,:,0]*np.cos(invariant_angle)+logChr[:,:,1]*np.sin(invariant_angle)
    I = np.exp(I)
    return I
            

            

if __name__ == "__main__":
    # Step1: Get image
    path = "/home/kishore/workspace/image_invaraince_entropy_minimization/shadow_test.jpg"
    img = cv2.imread(path)
    cv2.imshow("Original Image",img)
    print("Okay")
    # print (img.dtype)