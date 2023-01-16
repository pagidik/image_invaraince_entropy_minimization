import math
import argparse
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

U = np.array([ [1/math.sqrt(2), -1/math.sqrt(2), 0], [1/math.sqrt(6), 1/math.sqrt(6), -2/math.sqrt(6)] ]) # orthogonal matrix

def ShannonEntropy(I, bandwidth = 1):
    nbins = round((np.max(I) - np.min(I)) / bandwidth)
    P = np.histogram(I, nbins)[0] / I.size
    P = P[P != 0]
    return -np.sum(P * np.log2(P))

def projectOntoPlane(Rho, orthMatrix):
    return Rho @ orthMatrix.T

def getProjectedImage(Rho, radAngle):
    Chi = projectOntoPlane(Rho, U)
    I = Chi[:,:,0] * np.cos(radAngle) + Chi[:,:,1] * np.sin(radAngle)
    I = np.exp(I)
    cv.imshow("Invariant image",I)

    return I

def convertToLogChromacitySpace(img):

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    B[B == 0] 
    G[G == 0] = 1.0
    R[R == 0] = 1.0
    logChr = np.zeros_like(img, dtype = np.float64)
    gMean = np.power((R * G * B), 1.0/3)
    logChr[:, :, 0] = np.log((B / gMean))
    logChr[:, :, 1] = np.log((G / gMean))
    logChr[:, :, 2] = np.log((R / gMean))
    logChrB = np.atleast_3d(logChr[:, :, 0])
    logChrG = np.atleast_3d(logChr[:, :, 1])
    logChrR = np.atleast_3d(logChr[:, :, 2])
    print(logChrB)
    Rho = np.concatenate((logChrR, logChrG, logChrB), axis = 2) # log chromaticity on a plane
    return Rho

def solveProjection(img, Rho, nangles = 181):
    
    N = img.shape[0] * img.shape[1]
    Chi = projectOntoPlane(Rho, U)

    radians = np.radians(np.linspace(0, 180, nangles))
    Entropies = np.zeros(nangles, dtype = np.float64)
    for i, rad in enumerate(radians):
        I = Chi[:,:,0] * np.cos(rad) + Chi[:,:,1] * np.sin(rad)
        IMean = np.mean(I)
        IStd = np.std(I)
        lbound = IMean + 3.0 * (-IStd)
        rbound = IMean + 3.0 * (+IStd)
        IClipped = np.clip(I, lbound, rbound)
        binWidth = 3.5 * IStd * N**(-1/3)
        Entropies[i] = ShannonEntropy(IClipped, binWidth)

    minEntropy = np.min(Entropies)
    minEntropyIdx = np.argmin(Entropies)
    minEntropyAngle = radians[minEntropyIdx]
    return minEntropy, minEntropyAngle, Entropies

def L1(img, Rho, minEntropyAngle):
    N = img.shape[0] * img.shape[1]
    Chi = projectOntoPlane(Rho, U)
    e = np.array([-1 * math.sin(minEntropyAngle), math.cos(minEntropyAngle)])
    eT = np.array([np.cos(minEntropyAngle), np.sin(minEntropyAngle)])
    Ptheta = np.outer(eT, eT)
    Chitheta = Chi @ Ptheta.T
    I = Chi @ e
    Itheta = Chitheta @ e

    IMostBrightest = np.sort( I.reshape(I.shape[0] * I.shape[1]) )
    IMostBrightest = IMostBrightest[IMostBrightest.size - int(0.01*math.ceil(N)) : IMostBrightest.size]
    IMostBrightestTheta = np.sort( Itheta.reshape(Itheta.shape[0] * Itheta.shape[1]) )
    IMostBrightestTheta = IMostBrightestTheta[IMostBrightestTheta.size - int(0.01*math.ceil(N)) : IMostBrightestTheta.size]
    ChiExtralight = (np.median(IMostBrightest) - np.median(IMostBrightestTheta)) * e
    Chitheta += ChiExtralight

    Rhoti = Chitheta @ U
    cti = np.exp(Rhoti)
    ctiSum = np.sum(cti, axis = 2)
    ctiSum = ctiSum.reshape(cti.shape[0], cti.shape[1], 1)
    rti = cti / ctiSum
   
    return rti
    

    # print(ChiExtralight)
    # exit();
    



if __name__ == "__main__":
    
    path  = "/home/kishore/workspace/image_invaraince_entropy_minimization/3.jpeg"
    # Prepare input
    img = cv.imread(path)
    img64 = cv.GaussianBlur(img.astype(np.float64), (3, 3), 1)

    # Infer intrinsic image
    Rho = convertToLogChromacitySpace(img64)
    cv.imshow("Log Chromaticity Original", cv.normalize(Rho, 0, 255, cv.NORM_MINMAX))
    
    minEntropy, minEntropyAngle, Entropies = solveProjection(img, Rho, nangles = 181)
    invariantAngle = minEntropyAngle + np.pi / 2 # invariantAngle = (158.0 * np.pi / 180.0)
    I1D = getProjectedImage(Rho, invariantAngle)

    R = L1(img64, Rho, minEntropyAngle)
    cv.imshow("I1D", cv.normalize(R, 0, 255, cv.NORM_MINMAX))
    cv.waitKey()
    # Display results

    print(">>> Min entropy: ", minEntropy)
    print(">>> Min entropy angle (deg): ", np.rad2deg(minEntropyAngle))

    Chi = projectOntoPlane(Rho, U)
    minX1, maxX1 = np.min(Chi[:,:,0]), np.max(Chi[:,:,0])
    minX2, maxX2 = np.min(Chi[:,:,1]), np.max(Chi[:,:,1])
    chi1Range = maxX1 - minX1
    chi2Range = maxX2 - minX2
    chiCenter = np.array([(maxX1 + minX1) / 2.0, (maxX2 + minX2) / 2.0])
    chiHypot = np.hypot(chi1Range, chi2Range)

    RotMatx = lambda rad : np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype = np.float32)
    def Shift(arrayXY, x, y):
        shiftX, shiftY = np.mean(arrayXY[0]) - x, np.mean(arrayXY[1]) - y
        arrayXY[0] -= shiftX
        arrayXY[1] -= shiftY

    lightnintDirection = np.array([[0, chiHypot], [0, 0]])
    lightnintDirection = RotMatx(minEntropyAngle) @ lightnintDirection
    Shift(lightnintDirection, chiCenter[0], chiCenter[1])
    
    invatiantDirection = np.array([[0, chiHypot], [0, 0]])
    invatiantDirection = RotMatx(invariantAngle) @ invatiantDirection
    Shift(invatiantDirection, maxX1, maxX2)

    # Rotated Log Chromacity Gaussian
    def GaussianPDF(x, mu, var):
        denom = (2 * np.pi * var)**.5
        num = np.exp(-(x - mu)**2 / (2 * var))
        return num / denom
    I = Chi[:,:,0] * np.cos(minEntropyAngle) + Chi[:,:,1] * np.sin(minEntropyAngle)
    IMean = np.mean(I)
    IStd = np.std(I)
    lbound = IMean + 3.0 * (-IStd)
    rbound = IMean + 3.0 * (+IStd)
    GaussXAxis = np.linspace(lbound, rbound, 100)
    GaussYAxis = GaussianPDF(GaussXAxis, IMean, IStd**2)
    GaussYAxis /= np.linalg.norm(GaussYAxis)
    ChiGauss = np.array([[GaussXAxis, GaussYAxis]]).squeeze(0)
    ChiGauss = RotMatx(invariantAngle + np.pi) @ ChiGauss
    Shift(ChiGauss, 1.5 * maxX1, 1.5 * maxX2)
    
    cv.imshow("I1D", cv.normalize(I1D, 0, 255, cv.NORM_MINMAX))
    cv.waitKey()