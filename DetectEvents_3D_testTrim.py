from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from obspy import UTCDateTime
import numpy as np
import sys,os
sys.path.insert(0, '/home/andrewr/Desktop/Python')
from Reynen_Functions import extractFeatureData,staInfo2Dict

# Function to calculate PS delay with a given hypocentral distance (all units in km, and seconds)
def calcPSarrivalTimes(Vp,Vs,Dp,Ds,hypDist):
    tp=(hypDist+Vp*Dp)/Vp
    ts=(hypDist+Vs*Ds)/Vs
    return tp,ts

# Calculate the hypocentral distance given a PS delay
def calcHypDist_viaDelayPS(Vp,Vs,Dp,Ds,delayPS):
    hyp=Vp*Vs*(delayPS+Dp-Ds)/(Vp-Vs)
    return hyp 
    
# Return the softmax value of a given 3D array, along the depth axis
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=2).reshape((x.shape[0],x.shape[1],1))

# Collect all of the wx terms (seperately for phase types, and event classifications)
def get_WX_Terms(w0File,numTimeSamples,numUnqVars,meanFile,stdevFile,stream):
    numFeatures=numTimeSamples*numUnqVars*2
    w0=np.load(w0File)
    xMean=np.load(meanFile)
    xStdev=np.load(stdevFile)
    nClass=len(w0[0])
    # Split the weights and normalization terms into those for P and those for S
    w0P,w0S=w0[:numFeatures/2,:],w0[numFeatures/2:,:]
    w0Split=[[w0P[:,i].reshape((numUnqVars,numTimeSamples)),w0S[:,i].reshape((numUnqVars,numTimeSamples))] for i in range(nClass)]
    xMeanP=xMean[:numFeatures/2].reshape((numUnqVars,numTimeSamples))
    xMeanS=xMean[numFeatures/2:].reshape((numUnqVars,numTimeSamples))
    xStdevP=xStdev[:numFeatures/2].reshape((numUnqVars,numTimeSamples))
    xStdevS=xStdev[numFeatures/2:].reshape((numUnqVars,numTimeSamples))
    # Get all of the wanted data
    unqStas=np.unique([aTrace.stats.station for aTrace in stream])
    wx_OutDict={} # Will contain station name, times, and the WX arrays
    for aSta in unqStas:
        staStream=stream.select(station=aSta)
        # Check to make sure that it has all of the required features
        nPts=[aTrace.stats.npts for aTrace in staStream]
        if len(staStream)!=numUnqVars or np.max(nPts)!=np.min(nPts):
#            print aSta,'didnt have all features'
            continue
        # Sort by channel name
        staStream.sort(keys=['channel'])
        # Create the time series
        featureArr=np.array([aTrace.data for aTrace in staStream])
        timeArr=(staStream[0].times()+staStream[0].stats.starttime.timestamp)[:-(numTimeSamples-1)]
        # These values have to be normalized, so have to add some terms to the arrays
        featureArr_forNorm=np.vstack((featureArr,np.ones(featureArr.shape)))
        w0Split_forNorm=[[np.vstack((w0Split[i][0]/xStdevP,-1*xMeanP*w0Split[i][0]/xStdevP)),
                          np.vstack((w0Split[i][1]/xStdevS,-1*xMeanS*w0Split[i][1]/xStdevS))] for i in range(nClass)]
        # Cross-correlate for each phase, and class (since can only fit once in the y-direction, only 1 row - take that)
        wx=[[correlate2d(featureArr_forNorm, w0Split_forNorm[i][0], boundary='symm', mode='valid')[0],
             correlate2d(featureArr_forNorm, w0Split_forNorm[i][1], boundary='symm', mode='valid')[0]] for i in range(nClass)]
        wx_OutDict[aSta]={'t':timeArr}
        for i in range(nClass):
            wx_OutDict[aSta]['wx_p'+str(i)]=wx[i][0]
            wx_OutDict[aSta]['wx_s'+str(i)]=wx[i][1]
    return wx_OutDict

# Scan through a wxDict to see at which origin times
# ... will return a 3D array for each station (in a dictionary) which gives the P-S delay versus origin time
# ... and the differenced probability (eveProb-noiseProb) that the TS-point is an event for each class
def get_StaEveProb_TS(wxDict,b0File,nPSmax):
    staEveProb={}
    b0=np.load(b0File)
    b0N,b0E,b0R=np.load(b0File)
    nClass=len(b0)
    # For each station
    for sta, aWXdict in wxDict.iteritems():
        nTime=len(aWXdict['t'])
        wxb=np.empty((nPSmax+1,nTime,3))
        # Set all of the wx_pZ terms
        wxP=np.dstack((np.reshape(aWXdict['wx_p'+str(k)],(1,len(aWXdict['wx_p'+str(k)]),1)) for k in range(nClass)))
        wxS=np.dstack((np.reshape(aWXdict['wx_s'+str(k)],(1,len(aWXdict['wx_s'+str(k)]),1)) for k in range(nClass)))
        # Add in the P-window wx values, as well as the bias terms
        wxb[:,:,:]=wxP+b0.reshape(1,1,len(b0))
        # Add in the S-window wx values
        j=np.ones(wxb.shape[1])*(nPSmax+1) # Determines the length used for adding S window (via PS-delay)
        j[-nPSmax:]-=np.arange(1,nPSmax+1)
        j=j.astype(int)
        for i in range(wxb.shape[1]):
            wxb[:j[i],i,:]+=wxS[0,i:i+j[i],:]
        # NaN out the positions which could not have both a P and S window present simulataneously
        for i in range(wxb.shape[1]):
            wxb[j[i]:,i,:]=np.nan
        # Apply softmax
        probs=softmax(wxb)
        # Save the probabilites to the output dictionary
        staEveProb[sta]={'t':aWXdict['t'],'probs':probs}
    return staEveProb
    
# Use the station file to assign station locations
def assignCoords(staProbDict,staFile):
    staInfo=np.genfromtxt(staFile,delimiter=',',dtype=str)
    staLocDict=staInfo2Dict(staInfo)
    for sta in staProbDict.keys():
        staProbDict[sta]['Loc']=staLocDict[sta]['Loc']
    return staProbDict

# Some distance weighting likely required...
def distWeightFunc(Dists):
    weights=np.ones(len(Dists))
    weights[np.where(Dists>50)]=-0.00333*Dists[np.where(Dists>50)]+1.1667 # Used for OK
    return weights

#dists=np.linspace(0,300,100)
#plt.plot(dists,distWeightFunc(dists),'r')
#plt.show()
#quit()

# Extract the probabilities that an event is real, if above certain percentage
def get_eveOriginProb(Vp,Vs,Dp,Ds,timeDelta,staProbDict,staProbMinDiff,arrivalSampleOffset,
                      staProbDict_2):
    outInfo=[] # Will contain origin time, hypocentral distance, event probability,
               # ... P-Time-index, PS-Delay-Index, and the station location
    stas=[]
    for aSta in staProbDict.keys():
        # Base which times are selected on the ratio...
        # ... but stack based on the value
        probs=staProbDict[aSta]['probs']
        ## Testing ##
        argMax_2=np.argmax(staProbDict_2[aSta]['probs'],axis=2)
        ## Testing ##
        probN,probE=probs[:,:,0],probs[:,:,1]
        probDiff=((probE-probN)+1)/2.0
        ## HERE ##
        rowArg,colArg=np.where((probDiff>staProbMinDiff))
        ## Testing ##
#        rowArg,colArg=np.where((probDiff>staProbMinDiff)&(argMax_2==0))
        ## Testing ##
        times=staProbDict[aSta]['t']
        staLoc=staProbDict[aSta]['Loc']
        for pIdx,psDelayIdx,aProb in zip(colArg,rowArg,probE[rowArg,colArg]):
            pArrivalTime=times[pIdx]
            delayPS=psDelayIdx*timeDelta
            # Calculate the hypocentral distance from the PS delay (do not allow a negative PS delay)
            hypDist=np.max([calcHypDist_viaDelayPS(Vp,Vs,Dp,Ds,np.max([delayPS,0])),0])
            # Calculate the origin time, have to account for the feature offset
            tp=calcPSarrivalTimes(Vp,Vs,Dp,Ds,hypDist)[0]
            originTime=pArrivalTime-tp+arrivalSampleOffset*timeDelta
            # Add to the output array
            outInfo.append([originTime,hypDist,aProb,pIdx,psDelayIdx]+list(staLoc))
            stas.append(aSta)
    outInfo=np.array(outInfo,dtype=float)
    # Apply some sort of distance weighting to these probabilities
    outInfo[:,2]*=distWeightFunc(outInfo[:,1])
    # Make this array a bit easier to understand
    originInfo={'t':outInfo[:,0],'sta':np.array(stas,dtype=str),'dist':outInfo[:,1],'probE':outInfo[:,2],
                'pIdx':outInfo[:,3].astype(int),'dIdx':outInfo[:,4].astype(int),'loc':outInfo[:,5:8]}
    return originInfo

# Make a subset of rings which can be used later to stack the probabilities (and another base for the ROI)
def get_stackRings(roi,maxHypDist,maxVertDist,maxDelayPS,
                   timeDelta,Vp,Vs,Dp,Ds):
    # Figure out which boundaries should be applied
    ringBounds=[]
    for aDelay in np.arange(0,maxDelayPS+timeDelta,timeDelta):
        aHyp=calcHypDist_viaDelayPS(Vp,Vs,Dp,Ds,aDelay)
        if aHyp<0:
            aHyp=0
        if aHyp in ringBounds:
            continue
        ringBounds.append(aHyp)
    halfRingWidth=(ringBounds[-1]-ringBounds[-2])/2.0
    ringBounds.append(halfRingWidth*2+ringBounds[-1])
    ringBounds=np.array(ringBounds,dtype=float)-halfRingWidth
    # Create the base map
    gridSpace=halfRingWidth/2.0 # Choose this ratio for wanted ring resolution
    baseX,baseY,baseZ=np.meshgrid(np.arange(roi[0],roi[1],gridSpace),
                                  np.arange(roi[2],roi[3],gridSpace),
                                  np.arange(roi[4],roi[5],gridSpace))
    baseProb=np.zeros(baseX.shape)
    # Create the station rings, centered on zero
    usedMaxDist=maxHypDist-maxHypDist%gridSpace+gridSpace
    usedMaxVert=maxVertDist-maxVertDist%gridSpace+gridSpace
    layX,layY,layZ=np.meshgrid(np.arange(-usedMaxDist,usedMaxDist+0.1*gridSpace,gridSpace),
                               np.arange(-usedMaxDist,usedMaxDist+0.1*gridSpace,gridSpace),
                               np.arange(-usedMaxVert,usedMaxVert+0.1*gridSpace,gridSpace))
    layD=(layX**2+layY**2+layZ**2)**0.5
    rings=[]
    for i in range(len(ringBounds)-1):
        rings.append(np.array((layD>=ringBounds[i])&(layD<ringBounds[i+1]),dtype=np.uint8))
    return rings,ringBounds,baseX,baseY,baseZ,baseProb,halfRingWidth
    
# Assign specific layer indicies to the originInfo, as well as how they would fit into the baseLayer
def assignLayerInfo(originInfo,ringLay,ringBounds,baseX,baseY,baseZ,baseProb):
    # Assign which layer
    layIdx=np.digitize(originInfo['dist'],ringBounds)-1
    originInfo['layIdx']=layIdx
    # Figure out all the shift/trim for each station
    shapeBase=baseX.shape
    shapeLay=ringLay[0].shape
    centerLayIdx=np.array(shapeLay)/2
    unqStas,staIdxs=np.unique(originInfo['sta'],return_index=True)
    staPosDict={}
    for aSta,aStaIdx in zip(unqStas,staIdxs):
        # Calculate where the station is centered on the base layer
        sX,sY,sZ=originInfo['loc'][aStaIdx]
        argPos=np.argmin((baseX.flatten()-sX)**2+(baseY.flatten()-sY)**2+(baseZ.flatten()-sZ)**2)
        xArg,yArg,zArg=np.unravel_index(argPos,shapeBase)
        # Figure out what bounds to use to trim
        # These naming conventions based on plan view of the rings
        topArg=np.min([(shapeBase[0])-xArg,centerLayIdx[0]])
        botArg=np.min([xArg,centerLayIdx[0]])
        rightArg=np.min([(shapeBase[1])-yArg,centerLayIdx[1]])
        leftArg=np.min([yArg,centerLayIdx[1]])
        outArg=np.min([(shapeBase[2])-zArg,centerLayIdx[2]])
        inArg=np.min([zArg,centerLayIdx[2]])
        # Contains the position within the base array, and how far up/down/left/right/out/in should be used
        staPosDict[aSta]=np.array([xArg,yArg,zArg,botArg,topArg,leftArg,rightArg,inArg,outArg],dtype=int)
    # Assign these to each origin entry
    gridShifts=[]
    for aSta in originInfo['sta']:
        gridShifts.append(staPosDict[aSta])
    originInfo['gridShift']=gridShifts
    return originInfo

# Make an interactive plot for this time window, showing the overlapping regions
def makeOriginCircleSlider(originInfo,roi,startTime,endTime,timeDelta,rings,base,baseSpacing,eventDetections):
    # Set up some global variables, and reformat the origins info
    global slideVal,depSlideVal   
    slideVal=0
    depSlideVal=0
    # Get the center index of the ring arrays
    lcx,lcy,lcz=np.array(rings[0].shape)/2 # Array length is odd, so this will yield the center index
    depIdxLen=base.shape[2]
    queryTimes=np.arange(startTime.timestamp,endTime.timestamp,timeDelta)
    # Plot the figure
    fig = plt.figure(figsize=(12,14))
    # Add in the main image
    axMain = fig.add_axes([0.1, 0.35, 0.75, 0.60])
    # roi given here represents edges of the base probability stack array (not original, as was slightly edited)
    extent=[roi[0]-0.5*baseSpacing,roi[1]+0.5*baseSpacing,roi[2]-0.5*baseSpacing,roi[3]+0.5*baseSpacing]
    img=axMain.imshow(base[:,:,0],origin='lower',extent=extent,vmin=0,vmax=6,interpolation='none')
    txt=axMain.text(0.5,0.9,'',transform=axMain.transAxes,fontsize=14,color='white')
    evePoint=axMain.scatter([],[],c='white',s=55)
    axMain.set_xlim(extent[0],extent[1])
    axMain.set_ylim(extent[2],extent[3])
    # Plot the stations to ensure that they are in center    
    unqStas,staIdxs=np.unique(originInfo['sta'],return_index=True)
    for aIdx in staIdxs:
        loc=originInfo['loc'][aIdx]
        axMain.plot(loc[0],loc[1],'g^')
    axMain.set_aspect('equal')
    # Add in the histogram of how many entries there are
    axHist = fig.add_axes([0.1, 0.05, 0.8, 0.15])
    binArgs=np.digitize(originInfo['t'],queryTimes)
    binCounts = np.array([len(originInfo['t'][binArgs == i]) for i in range(1, len(queryTimes))])
    binUnqCounts = np.array([len(np.unique(originInfo['sta'][binArgs == i])) for i in range(1, len(queryTimes))])
    axHist.plot((queryTimes[1:]+queryTimes[:-1])/2.0,binCounts,'r')
    axHist.plot((queryTimes[1:]+queryTimes[:-1])/2.0,binUnqCounts,'b')
    axHist.set_ylim(0,np.max(binCounts))
    axHist.set_xlim(queryTimes[0],queryTimes[-1])
    axLine=axHist.plot([queryTimes[slideVal],queryTimes[slideVal]],[0,np.max(binCounts)],color='k',lw=2.5)[0]
    # Plot the event times
    for aEveTime in eventDetections['t']:
        axHist.axvline(aEveTime,color='g')
    # Add in the slider
    axSlider = fig.add_axes([0.1, 0.25, 0.8, 0.05])
    axDepSlider = fig.add_axes([0.90, 0.35, 0.07, 0.05])
    slider=Slider(axSlider,'Time Idx',0,len(queryTimes)-1,valinit=1,valfmt='%i')
    depSlider=Slider(axDepSlider,'Depth Idx',0,depIdxLen-1,valinit=0,valfmt='%i')
    # Set the slider value upon key press
    def keyPress(event):
        global slideVal,depSlideVal
        origSlideVal=slideVal
        origDepSlideVal=depSlideVal
        if event.key not in ['left','right','up','down']:
            return
        if event.key=='left' and slideVal>0:
            slideVal-=1
        elif event.key=='right' and slideVal<len(queryTimes)-1:
            slideVal+=1
        elif event.key=='up' and depSlideVal>0:
            depSlideVal-=1
        elif event.key=='down' and depSlideVal<depIdxLen-1:
            depSlideVal+=1
        if origSlideVal!=slideVal:
            slider.set_val(slideVal)
        if origDepSlideVal!=depSlideVal:
            depSlider.set_val(depSlideVal)
    # Update the image when the slider value changes
    def updateFrame(tArg):
        global slideVal
        slideVal=int(tArg)
        t1,t2=queryTimes[slideVal],queryTimes[slideVal+1]
        wantArgs=np.where((originInfo['t']>t1)&(originInfo['t']<=t2))[0]
        wantEveArgs=np.where((eventDetections['t']>t1)&(eventDetections['t']<=t2))[0]
        evePoint.set_offsets(eventDetections['loc'][wantEveArgs][:,:2])
        stack=np.copy(base)
        for anArg in wantArgs:
            # p has entries: cx,cy,cz,bot,top,left,right,in,out
            p,layIdx,prob=originInfo['gridShift'][anArg],originInfo['layIdx'][anArg],originInfo['probE'][anArg]
            stack[p[0]-p[3]:p[0]+p[4],p[1]-p[5]:p[1]+p[6],p[2]-p[7]:p[2]+p[8]]+=(
                prob*rings[layIdx][lcx-p[3]:lcx+p[4],lcy-p[5]:lcy+p[6],lcz-p[7]:lcz+p[8]])
        img.set_data(stack[:,:,depSlideVal])
        tCenter=UTCDateTime((t1+t2)/2.0)
        txt.set_text(str(tCenter)+'\nMaxProb: '+'{:.3f}'.format(np.max(stack[:,:,int(depSlideVal)])))
        axLine.set_data([tCenter,tCenter],[0,np.max(binCounts)])
        fig.canvas.draw()
    def pingSlider(depSlideArg):
        global depSlideVal
        depSlideVal=depSlideArg
        updateFrame(slideVal)
    slider.on_changed(updateFrame)
    depSlider.on_changed(pingSlider)
    fig.canvas.mpl_connect('key_press_event',keyPress)
    plt.show()

def detectEvents(originInfo,rings,baseX,baseY,baseZ,base,startTime,endTime,
                 timeDelta,featAvgIdxLen,maxTT,eveProbMinSum,Vp,Vs,Dp,Ds):
    outTimes,outLocs,outProbs,outArgs=[],[],[],[]
    # Get the center index of the ring arrays
    lcx,lcy,lcz=np.array(rings[0].shape)/2 # Array length is odd, so this will yield the center index
    # Make a list of all timestamps, with the feature interval (timeDelta) as a spacing
    queryTimes=np.arange(startTime.timestamp,endTime.timestamp,timeDelta)
    # Make list of stations which are used in this time range, and a reference index for each station
    segmentStas=np.unique(originInfo['sta'])
    refInfo={}
    for aSta in segmentStas:
        refIdx=list(originInfo['sta']).index(aSta)
        tp,ts=calcPSarrivalTimes(Vp,Vs,Dp,Ds,originInfo['dist'][refIdx])
        refpIdx=originInfo['pIdx'][refIdx]-int(np.round(tp/timeDelta))
        refOrig=originInfo['t'][refIdx]-int(np.round(tp/timeDelta))
        refInfo[aSta]=[refIdx,refOrig,refpIdx]
    # Now scan through all of these times, removing the duplicates...
    # ... each pass the maximums over a sweeping window of length "maxTT" is extracted, and any indices on the two diagonals...
    # ... and two verticals are removed from the other potential events - and then their probability is updated
    search=True
    seenArgs=np.zeros(len(originInfo['t']))
    while search:
        # For each segment of times, find the location of the maximum probability (and get its index)
        eveProbs,eveLocs,eveTimes,usedOrigArgs=[],[],[],[]
        for i in range(0,len(queryTimes)-1):
            t1,t2=queryTimes[i],queryTimes[i+1]
            aEveTime=(t1+t2)/2.0
            wantArgs=np.where((originInfo['t']>t1)&(originInfo['t']<=t2)&(seenArgs==0))[0]
            stack=np.copy(base)
            for anArg in wantArgs:
                # p has entries: cx,cy,cz,bot,top,left,right,in,out
                p,layIdx,prob=originInfo['gridShift'][anArg],originInfo['layIdx'][anArg],originInfo['probE'][anArg]
                stack[p[0]-p[3]:p[0]+p[4],p[1]-p[5]:p[1]+p[6],p[2]-p[7]:p[2]+p[8]]+=(
                    prob*rings[layIdx][lcx-p[3]:lcx+p[4],lcy-p[5]:lcy+p[6],lcz-p[7]:lcz+p[8]])
            argMaxX,argMaxY,argMaxZ=np.unravel_index(np.argmax(stack.flatten()),stack.shape)
            aEveProb=stack[argMaxX,argMaxY,argMaxZ]
            # If there were no events during this time, skip
            if aEveProb<eveProbMinSum:
                continue
            # If there was more than one entry at this value, then get the median position
            if len(np.where(stack==aEveProb)[0])>1:
                argXs,argYs,argZs=np.unravel_index(np.where(stack.flatten()==aEveProb)[0],stack.shape)
                # Take median of index
                argMaxX,argMaxY,argMaxZ=int(np.round(np.median(argXs))),int(np.round(np.median(argYs))),int(np.round(np.median(argZs)))
                # In the case where the median is outside the arg max domain, take first argument
                if stack[argMaxX,argMaxY,argMaxZ]<aEveProb:
                    argMaxX,argMaxY,argMaxZ=argXs[0],argYs[0],argZs[0]
            aEveX,aEveY,aEveZ=baseX[argMaxX,argMaxY,argMaxZ],baseY[argMaxX,argMaxY,argMaxZ],baseZ[argMaxX,argMaxY,argMaxZ]
            # Given this maximum location, figure out which originInfo rows were used
            aUsedOrigArg=[]
            for anArg in wantArgs:
                p,layIdx=originInfo['gridShift'][anArg],originInfo['layIdx'][anArg]
                # If the ring layers do not cover the entire stack area... 
                # ...check first to see if current ring could stack at the potential event location
                cutRing=rings[layIdx][lcx-p[3]:lcx+p[4],lcy-p[5]:lcy+p[6],lcz-p[7]:lcz+p[8]]
                if np.sum(np.array([argMaxX,argMaxY,argMaxZ])>=cutRing.shape)>0:
                    continue
                # ... then check if used at potential event location.
                if cutRing[argMaxX-(p[0]-p[3]),argMaxY-(p[1]-p[5]),argMaxZ-(p[2]-p[7])]==1:
                    aUsedOrigArg.append(anArg)
            usedOrigArgs.append(np.array(aUsedOrigArg,dtype=int))
            eveProbs.append(aEveProb)
            eveLocs.append([aEveX,aEveY,aEveZ])
            eveTimes.append(aEveTime)
        # If there are no more peaks to review, break
        if len(eveProbs)==0:
            search=False
            break
        eveProbs,eveLocs,eveTimes=np.array(eveProbs),np.array(eveLocs),np.array(eveTimes)
        # Collect all maximums, with the +/- maxTT bounding them
        collectPeaks=True
        peakIdxs=[]
        copyEveProbs=np.copy(eveProbs) # Used as a padding array for this "search" iteration
        while collectPeaks:
            maxArg=np.argmax(copyEveProbs)
            # If the peak was seen, swap it and any nearby peaks to be zero
            nearbyArgs=np.where(np.abs(eveTimes-eveTimes[maxArg])<maxTT)
            copyEveProbs[nearbyArgs]=0
            if np.max(copyEveProbs)<eveProbMinSum:
                collectPeaks=False
            # This peak however still requires to be a local maximum (in the unaltered array)...
            # ... as to not ruin the chance of a larger peak close to an even larger peak (which... 
            # ... was already added to peakIdxs during this "search" iteration)
            if eveProbs[maxArg]<np.max(eveProbs[nearbyArgs]):
                continue
            peakIdxs.append(maxArg)
        peakIdxs=np.array(peakIdxs)
        # For each peak, look at the surrounding origin entries to see if they should be ignored later
        for aPeakArg in peakIdxs:
            # Collect nearby (in time) origin information (no need to scan all origin entries)
            aEveTime=eveTimes[aPeakArg]
            aEveLoc=eveLocs[aPeakArg]
            oArgs=np.where(np.abs(aEveTime-originInfo['t'])<maxTT)[0]
            # Collect the diagonal and vertical value from the peak for each station...
            peakDiags=[originInfo['pIdx'][anArg]+originInfo['dIdx'][anArg] for anArg in usedOrigArgs[aPeakArg]]
            peakVerts=[originInfo['pIdx'][anArg] for anArg in usedOrigArgs[aPeakArg]]
            peakStas=[originInfo['sta'][anArg] for anArg in usedOrigArgs[aPeakArg]]
            # ... including the theoretical peakDiags, and peakVerts if the station was not helpful in the stack
            for aSta in [sta for sta in segmentStas if sta not in peakStas]:
                # Find out what the pIdx would have been
                refIdx,refOrig,refpIdx=refInfo[aSta] # Get the reference origin time and pIdx (dIdx=0)
                staLoc=originInfo['loc'][refIdx]     # Calculate the P and S travel times
                aDist=np.sum((aEveLoc-staLoc)**2)**0.5
                tp,ts=calcPSarrivalTimes(Vp,Vs,Dp,Ds,aDist)
                psDelayIdx=int(np.round((ts-tp)/timeDelta))
                pIdx=int(np.round((aEveTime+tp-refOrig)/timeDelta))+refpIdx # Get P-index relative to reference
                peakDiags.append(pIdx+psDelayIdx)
                peakVerts.append(pIdx)
                peakStas.append(aSta)
            peakDiags,peakVerts,peakStas=np.array(peakDiags),np.array(peakVerts),np.array(peakStas)
            # Collect the diagonals and vertical index values from other times near the peak...
            # ... For varying S-wave (fix-P), and S-weights being high on P-wave
            oDiags=np.array([originInfo['pIdx'][anArg]+originInfo['dIdx'][anArg] for anArg in oArgs])
            # ... For varying P-wave (fix-S), and P-weights being high on S-wave
            oVerts=np.array([originInfo['pIdx'][anArg] for anArg in oArgs])
            oStas=np.array([originInfo['sta'][anArg] for anArg in oArgs])
            # For each station... (should only be one observation per station for a given peak - as rings do not overlap)
            for i,aSta in enumerate(peakStas):
                # ...Look at the difference between the diagonals  
                peakDiag,peakVert=peakDiags[i],peakVerts[i]
                staDupArgs=(np.where((oStas==aSta)&
                                     ((np.abs(oDiags-peakDiag)<=(featAvgIdxLen))|      # Varying S-wave (fix P)
                                      (np.abs(oVerts-peakDiag)<=(featAvgIdxLen))|      # P-weights being high on S-wave
                                      (np.abs(oVerts-peakVert)<=(featAvgIdxLen))|      # Varying P-wave (fix-S)
                                      (np.abs(oDiags-peakVert)<=(featAvgIdxLen)))))[0] # S-weights being high on P-wave
                # ...convert back to the main arrays indices, and set them as "seen"
                seenArgs[oArgs[staDupArgs]]=1
            # Finally add this information to the event detections
            outTimes.append(eveTimes[aPeakArg])
            outLocs.append(eveLocs[aPeakArg])
            outProbs.append(eveProbs[aPeakArg])
            outArgs.append(usedOrigArgs[aPeakArg])
    eveDetections={'t':np.array(outTimes),'loc':np.array(outLocs),'prob':np.array(outProbs),'originArgs':outArgs}
    return eveDetections

# Output the events to a pick file
def writeEvents(eveDetections,originInfo,startEveIdx,outSumDir,outPickDir,
                startTime,endTime,intervalBuff,arrivalSampleOffset,timeDelta,
                Vp,Vs,Dp,Ds):
    outName=startTime.strftime('%Y%m%d.%H%M%S')
    # Sort by event time
    sortArg=np.argsort(eveDetections['t'])
    sumArr=[]
    # For each origin, make an entry for its
    for aID,arg in enumerate(sortArg):
        eveTime=UTCDateTime(eveDetections['t'][arg])
        # Skip if this event is within the buffer
        if eveTime<startTime+intervalBuff/2.0 or eveTime>=endTime-intervalBuff/2.0:
            continue
        # Add to the summary
        sumArr.append([str(aID+startEveIdx).zfill(6)]+list(eveDetections['loc'][arg])+['-999',
                       str(eveTime),'{:.3f}'.format(eveDetections['prob'][arg]),
                       str(len(eveDetections['originArgs'][arg]))])
        # Write a new picks file
        picks=[]
        for oArg in eveDetections['originArgs'][arg]:
            sta,hypDist=originInfo['sta'][oArg],originInfo['dist'][oArg]
            tp,ts=calcPSarrivalTimes(Vp,Vs,Dp,Ds,hypDist)
            picks.append([sta,'P',(eveTime+tp).timestamp])
            picks.append([sta,'S',(eveTime+ts).timestamp])
        np.savetxt(outPickDir+'/'+str(aID+startEveIdx).zfill(6)+eveTime.strftime('_%Y%m%d.%H%M%S.%f.picks'),
                   np.array(picks,dtype=str),fmt='%s',delimiter=',')
    np.savetxt(outSumDir+'/'+outName+'_Summary.csv',np.array(sumArr),fmt='%s',delimiter=',',
               header='ID,X,Y,X,Mag,DateTime,EveSumProb,NumSta')


def main():
    archiveDir='../EventDetection/FeatureArchive_Continuous'
    firstTime='2015-09-08 12:00:00' # Start of processing time
    lastTime='2015-09-08 12:30:00'
    interval=2700.0 # Interval used in extracting locations
    startEveIdx=16020 # Starting event index
    w0File='../EventDetection/W0.npy'
    b0File='../EventDetection/B0.npy'
    meanFile='../EventDetection/xMean.npy'
    stdevFile='../EventDetection/xStdev.npy'
    ## Testing ##
    w0File_2='./stack_W0.npy'
    b0File_2='./stack_B0.npy'
    meanFile_2='./stack_xMean.npy'
    stdevFile_2='./stack_xStdev.npy'
    ## Testing ##
    staFile='../EventDetection/NX_stations_UTM14.csv'
    outSumDir='./NewDetectionSummaries/3D_2017May17_lowTsum'
    outPickDir='./NewDetectionPicks/3D_2017May17_lowTsum'
    numTimeSamples=8 # How many time samples are used during classification
    arrivalSampleOffset=1 # How many samples before arrival time are used during classification
    timeDelta=1.0 # How often the features are sampled
    featAvgIdxLen=6 # How many samples features are averaged over (used for removing high values beside peaks)
    numUnqVars=39 # How many different unique variables there are in the features
    Vp,Vs=6.17,3.55
    Dp,Ds=-0.71,-0.97
    maxVertDist=20 # Max vertical distance allowed on any shell
    maxHypDist=200 # Max hypocentral distance allowed on any shell
    staProbMinDiff=0.3 # Probability difference on a per station basis, if observation is used (Normalized between 0 and 1)
    eveProbMinSum=1.00 #ex. for 2.0, require 4 stations with atleast 0.5 prob (averaged)
    roi=[540, 770,3900, 4120, -1.0, 14.0] # Region of interest - x1,x2,y1,y2,z1,z2
    
    # Generate the output folders
    for aDir in [outSumDir,outPickDir]:
        if not os.path.exists(aDir):
            os.makedirs(aDir)
    
    
    # Specify looping parameters...
    firstTime,lastTime=UTCDateTime(firstTime).timestamp,UTCDateTime(lastTime).timestamp
    # Calculate the max duration a P and S arrival can differ (over the desired distance range)
    PSdelay=np.diff(calcPSarrivalTimes(Vp,Vs,Dp,Ds,maxHypDist))[0]
    maxSampShift=int(np.ceil(PSdelay/timeDelta))
    # Calculate the maximum time between origins of... 
    # ...least overlap of two "V's" - seen in the pIdx vs dIdx probability plots
    maxTT=1*calcPSarrivalTimes(Vp,Vs,Dp,Ds,maxHypDist)[0]+2*maxSampShift*timeDelta
    intervalBuff=maxTT*2.1
    # Figure out what ranges should be used for the rings
    ringLay,ringBounds,baseX,baseY,baseZ,baseProb,distErr=get_stackRings(roi,maxHypDist,maxVertDist,PSdelay,
                                                                         timeDelta,Vp,Vs,Dp,Ds)
    # Loop through all times
    for t1 in np.arange(firstTime,lastTime,interval):
        startTime,endTime=UTCDateTime(t1),UTCDateTime(t1+interval+intervalBuff)
        print startTime,endTime
        # Get the wx dictionary, for each phase and classification type
        stream=extractFeatureData(archiveDir,startTime,endTime,[['*','*']],fillVal=np.nan)
        wxDict=get_WX_Terms(w0File,numTimeSamples,numUnqVars,meanFile,stdevFile,stream)
        ## Testing ##
        wxDict_2=get_WX_Terms(w0File_2,numTimeSamples,numUnqVars,meanFile_2,stdevFile_2,stream)
        staProbDict_2=get_StaEveProb_TS(wxDict_2,b0File_2,maxSampShift)
        ## Testing ##
        # Get the probabilities for each classification, in terms of P-arrival, and PS delay
        staProbDict=get_StaEveProb_TS(wxDict,b0File,maxSampShift)
        staProbDict=assignCoords(staProbDict,staFile)
        # Collect all of the probability ratios above the desired per-station threshold
        originInfo=get_eveOriginProb(Vp,Vs,Dp,Ds,timeDelta,staProbDict,staProbMinDiff,arrivalSampleOffset,
                                     staProbDict_2)
        # Assign information the ring index, and station grid bounds (for stacking)
        originInfo=assignLayerInfo(originInfo,ringLay,ringBounds,baseX,baseY,baseZ,baseProb)
        # Detect events 
        eveDetections=detectEvents(originInfo,ringLay,baseX,baseY,baseZ,baseProb,startTime,endTime,
                                   timeDelta,featAvgIdxLen,maxTT,eveProbMinSum,Vp,Vs,Dp,Ds) 
        print len(eveDetections['t']),'events detected...'
        # Write out the events
        writeEvents(eveDetections,originInfo,startEveIdx,outSumDir,outPickDir,
                    startTime,endTime,intervalBuff,arrivalSampleOffset,timeDelta,
                    Vp,Vs,Dp,Ds)
        startEveIdx+=len(eveDetections['t'])
#        # Show the event probabilities over time, interactively
#        baseSpacing=baseX[0][1][0]-baseX[0][0][0]
#        makeOriginCircleSlider(originInfo,[np.min(baseX),np.max(baseX),np.min(baseY),np.max(baseY),np.min(baseZ),np.max(baseZ)],
#                                           startTime,endTime,timeDelta,ringLay,baseProb,baseSpacing,eveDetections)


if __name__ == "__main__":
    main()
