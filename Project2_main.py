import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from Project2_self import chrom_case_a,chrom_case_b,chrom_case_c, get_log_chrom, projections_case_a, invariant_image_case_a,projections,invariant_image
import Project2_self

'''
Find the grayscale of the float64 converted image
'''

if __name__ == "__main__":

    # Step1: Get image
    path = "/home/kishore/workspace/image_invaraince_entropy_minimization/3.jpeg"
    # path = "/Users/sumeghasinghania/Downloads/macbeth.png"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img64 = cv2.GaussianBlur(img.astype(np.float64), (3, 3),1 )
    cv2.imshow("Original Image",img)
    cv2.imshow("Guassian Image",img64)
    print(img[0][0][0].dtype)

    case = 'C'

    if (case=='A'):
        c = input ("Enter color channel: B/G/R\n")
        chromaticity = chrom_case_a(img,c)
        log_chromaticity,log_points = get_log_chrom(chromaticity)
        invariant_angle = projections_case_a(img,log_chromaticity,c)
        invariant_img = invariant_image_case_a(log_chromaticity,invariant_angle,c)
        
    elif(case=='B'):
        chromaticity = chrom_case_b(img)
        log_chromaticity,log_points = get_log_chrom(chromaticity)
        invariant_angle = projections(img,log_points)
        invariant_img = invariant_image(log_points,invariant_angle)

    else:
        chromaticity = chrom_case_c(img64)
        log_chromaticity,log_points = get_log_chrom(chromaticity)
        invariant_angle = projections(img,log_points)
        invariant_img = invariant_image(log_points,invariant_angle)

    cv2.waitKey()

