# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:24:36 2021

@author: iox36199
"""



import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.transform import *
from skimage.data import shepp_logan_phantom, checkerboard
from skimage import data
from skimage.data import *
import time
from approximating_fibre_area import main as approx_area
import pickle as pkl
rng = np.random.default_rng()

def initialEstimate(verticalReadout):
    """
    Given that we assume a uniform image, we have 3 projections, we return the guess of what our distribution should be. 
    """
     #3 projections
    
    
    oneD = int(len(verticalReadout))
    num_Cells = oneD**2
    est_1 = np.mean(np.array(verticalReadout)/oneD)  #Estimate that it's uniform 
    dimensions = (oneD,oneD)
    initialGuess = np.array([est_1]*oneD**2).reshape(dimensions)
    return initialGuess


def firstReconstruction(sinograms,angles):
    angles = np.sort(angles)
    firstImage = iradon(sinograms,theta=angles)
    return firstImage

def divide(a,b):
    c = np.divide(a,b,out=np.zeros_like(a), where=b!=0)
        
        
    return c
        
def findBP(myTemplates):
    BP = np.zeros_like(myTemplates[0]) #Take first template size
    angles  = len(myTemplates)
    dim = myTemplates[0].shape
    for projection in myTemplates:
        BP+=projection/(dim[0]*angles)
        
    return BP
def uniform(myTemplates):
    summedStrip = myTemplates[0] #This assumes the first angle is 0 degrees 
    dim = myTemplates[0].shape[0]
    average = summedStrip/dim
    uniformArray = np.array([average]*dim**2).reshape(dim,dim)
    return uniformArray


# def iRadonStart(myTemplates,sinograms):
    
def saveRatiosAndPositions(ratios,indexes):
    with open('ratioData.pkl','wb') as file:
        pkl.dump(ratios,file)
    with open('indexData.pkl','wb') as file2:
        pkl.dump(indexes,file2)
    
def loadRatiosAndPositions():
    with open('ratioData.pkl','rb') as file:
        ratios = pkl.load(file)
    with open('indexData.pkl','rb') as file2:
        indexes = pkl.load(file2)     #This will load up ratio and index data for panels: 0, 45,65,25 (in that order)
        return ratios,indexes

def MLEM_rotate(im,angles,attenuated): #START 
    """
    im: Original image we attempt to recreate
    angles: List of angles in degrees from which we are measuring from, 0 deg being in the +X axis direction -->
    attenuated:
    """
    
    
    projections = len(angles) #How many angles there are
    n = im.shape[0] #One dimension of the image.
    sinogram3dDimensions = (len(angles),n,n) #Shape of data storage (image for each projection)
    attenuationMasks = np.zeros( sinogram3dDimensions) #attenuation
    myTemplates = np.zeros( sinogram3dDimensions)
    myGuesses = np.zeros( sinogram3dDimensions)
    ratioLists = []
    fibreIndexes = []
    
    # for i in range(projections):
    #     thisRatioList,thisFibreIndex = approx_area(1000,angles[i])
    #     ratioLists.append(thisRatioList)
    #     fibreIndexes.append(thisFibreIndex)  # only uncomment if new angles are used
        
        #For every angle, find the ratio of pixel to fibre area per pixel, and the respective indices
    #saveRatiosAndPositions(ratioLists,fibreIndexes)
    
    ratioLists,fibreIndexes = loadRatiosAndPositions()
    for i in range(projections):
        myTemplates[i], attenuationMasks[i] = makeSqTemp(n,angles[i],im,attenuated,angles,ratioLists[i],fibreIndexes[i])[:2]
    mySinogram,sinoIm = makeSinogram(im,angles,attenuated)
    
    initialGuess = firstReconstruction(mySinogram[:,0,:].T,np.sort(angles))
    for i in range(projections):
        myGuesses[i] = makeSqTemp(n,angles[i],initialGuess,attenuated,angles,ratioLists[i],fibreIndexes[i])[0]
    
    
    iterGuess = initialGuess
    measuredValues,guessingValues,attValues = [np.zeros((projections))]*3
    storeRatio = np.zeros((n,n))
    ratioDifferenceList = []
    stdDevList = []
    stdDev = 1
    stdDevOld = 100
    avgRatio = 50
    iteration = 0
    start = time.time()
    for i in range(50):
        previousGuess = iterGuess
        stdDevOld = stdDev
        iteration+=1
        for i in range(n):
            
            for j in range(n):
                
                measuredValues,guessingValues = myTemplates[:,i,j],myGuesses[:,i,j]
                MG = divide(measuredValues,guessingValues) #Divide the measured values from the guessed value in an array
                ratio = divide(np.sum(MG),len(MG))
                storeRatio[i,j] = ratio
                #print(attValues)
                iterGuess[i,j] = iterGuess[i,j]*ratio

        print(f'Iteration = {iteration}')
        
        stdDev = deviation(iterGuess,im)    
        
        stdDevList.append(stdDev)
        avgRatio = np.mean(storeRatio)
        ratioDifferenceList.append(avgRatio)
        print(f'Std = {stdDev}')
        print(f'Avg. Ratio = {avgRatio}%')
        for s in range(projections):
            
            myGuesses[s] = makeSqTemp(n,angles[s],iterGuess,attenuated,angles,ratioLists[s],fibreIndexes[s])[0]
    print('\n')
    print(f'Duration: {time.time()-start}')
    fig = plt.figure()
    plt.imshow(previousGuess)
    print(f'Final Iteration: {iteration-1}')
    print(f'Final Std: {stdDevList[-2]}')
    print(f'Final Avg. Ratio: {ratioDifferenceList[-2]}')
    return previousGuess,stdDevList,ratioDifferenceList
    

def deviation(myGuess,original):
    """
    The coefficient of variation gives us a bog-standard way of analysying the improvements ML-EM is doing, however the low values
    (e.g 3.8% different) does not represent the visual "likeness" of the picture, and so users should only use this as a comparison to other
    variation values, not the absolute value. 
    """
    
    sqDif = (myGuess - original)**2
    variance = np.mean(sqDif)
    
    return np.sqrt(variance)
    

    

def makeSqTemp(n,angle,comparisonMatrix,attenuated,angles,currentRatioList,fibreIndex):
    """
    This function aims to reproduce the summed values of the comparisonMatrix 2D array WHICH INCLUDES accounting for overlaps in surface area of more than one fibre per fibre. This array can be either our iterated guess image, or our measured image. 
    It forward projects the sum, line 171 attenuates each pixel depending on its position. readoutMatrix is the output, and is a form in which we represent our forward projected readouts (so that i,j on the projected readout corresponds to the i,j on our iterated Guess for angle a).
    
    """
    
    a = np.deg2rad(angle)

    x,y = np.arange(0,n),np.arange(0,n) #Needs to be length n+1 otherwise columns/rows are lost through rotation
    X,Y = np.meshgrid(x,y) #A meshgrid to indicate positions on the 2D image
    midIndex = round((n+1)/2)
    
    xpr,ypr = X-midIndex,Y-midIndex
    cos_a,sin_a = np.cos(a),np.sin(a)
    
    
    rotatedY = np.round(xpr*sin_a - ypr*cos_a)
    
    rotatedX = np.round(xpr*cos_a + ypr*sin_a)   #Rotated X and Y coordinates
    readoutMatrix = np.zeros((n,n))
    ratioArray = np.zeros((n,n))
    
    attenuationMask = np.zeros((n,n))
    mySinogram = []
    sino = np.zeros(100)
    if np.min(np.round(rotatedY))==-n//2:
        rotatedY +=1
    rotatedY = np.round(rotatedY)
    stopping = round(n/2)+1 #Iterate from the first to last line at an angle a, where the midpoint line intersects with the origin
    starting= -round(n/2)+1
    count = 0
    for i in range(starting,stopping):
        pX,pY = fibreIndex[count][0],fibreIndex[count][1] #These are all the voxels that overlap with our fibre of interest
        removeEdge = np.where((pX ==n) | (pY==n)) 
        pX, pY = np.delete(pX,removeEdge),np.delete(pY,removeEdge)
        #Meshgrid produces an extra colum+row which isn't on our original image
        #Not sure if it's necessary yet
        
        myRxs = rotatedX[pX,pY] #Find the fibre voxels in the perpendicular direction
        ratioArray = np.zeros((100,100))
        ratioArray[pX,pY] = currentRatioList[count]
        if attenuated==True:
            attenuation = attenuate(myRxs,n,pX,pY)
            attenuationMask[pX,pY] = attenuation        
            readoutMatrix[pX,pY] += np.sum(comparisonMatrix[pX,pY]*ratioArray[pX,pY])*ratioArray[pX,pY]
            sino[count] = np.sum(comparisonMatrix*ratioArray)
        else:
            readoutMatrix[pX,pY] += np.sum(comparisonMatrix[pX,pY]*np.array(currentRatioList))
        count+=1
    
    print(len(mySinogram))
    return readoutMatrix,attenuationMask,sino

def attenuate(rX,n,pX,pY):
    d = n-rX #Arbitrary distance d from 399 (last index)
    weight = np.exp(-d/50)
    
    return weight
    

def evaluateRatioArray():
    ratioArrays = np.zeros((16,100,100))
    currentRatio,fibreIndex = loadRatiosAndPositions()
    for angle in range(len(currentRatio)):
        ratioArray = np.zeros((100,100))
        for fibre in range(len(currentRatio[0])):
            pX,pY = fibreIndex[angle][fibre][0],fibreIndex[angle][fibre][1]
            ratioArray[pX,pY] += currentRatio[angle][fibre]
        #assert (ratioArray.all()==1),"Ratio Array has a >1 value"
        ratioArrays[angle]=ratioArray
    return ratioArrays
        

def reorganiseSinogram(sinograms,angles):
    sinogramDict = dict(zip(angles,sinograms)) #Creates a key/item relationship between the angles and the respective measurements
    
    sortedAngles = sorted(sinogramDict) #Sorts the angles into a list
    sortedSinogram = np.array([sinogramDict[angle] for angle in sortedAngles]) #Append the measurements in order of the now sorted angles
    return sortedSinogram,sortedAngles

def makeSinogram(im,angles,attenuated):
    dim = im.shape[0]
    myArrays= np.zeros((len(angles),dim,dim)) #expects square image
    sinograms = np.zeros((len(angles),1,dim))
    sinogramLengths = []
    ratioLists,fibreIndexes = loadRatiosAndPositions()
    for i in range(len(angles)):
        
        mySinogram = makeSqTemp(len(im),angles[i],im,attenuated,angles,ratioLists[i],fibreIndexes[i])[2] #Need fibre stuff
        sinograms[i] = mySinogram
        
    
    sortedSinogram,sortedAngles = reorganiseSinogram(sinograms,angles) #sort the sinogram values based on their angles
    sortedSinogram = sortedSinogram[:,0,:].T
    sinoIm = extendSinogram(sortedSinogram,sortedAngles,dim)
    
    radonFP = radon(im,theta=np.array(sortedAngles)+180) #Produced radon transform from skimage
    radonFP = extendSinogram(radonFP,sortedAngles,dim)
    normComparison = np.max(radonFP)/np.max(sinoIm)
    sinoIm *=normComparison
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].imshow(sinoIm)
    
    normComparison = np.max(radonFP)/np.max(sinoIm)
    sinoIm *=normComparison
    
    ax[1].imshow(radonFP)
    ax[1].set_xlabel('Angle (deg)')
    fig.text(0.06,0.5,'Displacement of Projection',ha='center',va='center',rotation='vertical')
    ax[0].set_yticks([0,25,50,75,99])
    ax[0].set_yticklabels([-50,-25,0,25,49])
    ax[0].set_title('Sinogram from our work')
    ax[1].set_title('Sinogram from Skimage')
    ax[2].imshow(np.abs(sinoIm-radonFP))
    return sortedSinogram,sinoIm
    
        
def extendSinogram(sortedSinogram,sortedAngles,dim):
    sinoIm = np.zeros((dim,360))
    for i,angle in enumerate(sortedAngles):
        if i == len(sortedAngles) -1:
            break
        nextAngle = int(sortedAngles[i+1])
        angle = int(angle)
        angleDiff = int(nextAngle - angle)
        sinoIm[:,angle:nextAngle] = np.repeat(sortedSinogram[:,i],angleDiff).reshape(dim,angleDiff) #Increase the thickness of our
    return sinoIm

def forwardProject(sinograms):
    ratioLists,fibreIndexes = loadRatiosAndPositions()
    bp = np.zeros((100,100))
    for i in range(len(fibreIndexes)):
        for j in range(len(fibreIndexes[i])):
            bp[fibreIndexes[i][j][0],fibreIndexes[i][j]][1] += sinograms[i]
    return bp
    
def pickleAndSave(seed,sinogram,BPRecon,iRadonRecon,uniformRecon):
        pass
        
    
def panel(start):
    incr = 360/4
    return np.arange(start,360,incr)
increment = 360/4


angles = np.concatenate((panel(0),panel(45),panel(65),panel(25)))




myImage = resize(shepp_logan_phantom(),(100,100))
# myImage = np.zeros((100,100))
# myImage[30:40,30:40] = 1

myarray2 = MLEM_rotate(myImage,angles,True)
#sinograms2,sinoIm2 = makeSinogram(myImage,angles,True)
