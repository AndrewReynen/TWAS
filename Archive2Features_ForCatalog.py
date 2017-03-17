import numpy as np
from obspy import read, UTCDateTime
from obspy import Stream as EmptyStream
from obspy import Trace
from obspy.core.trace import Stats as TraceStats
import sys,os
sys.path.insert(0, '/home/andrewr/Desktop/Python')
from Reynen_Functions import getDOP,getDOPStats,getSonogram,staInfo2Dict,RemoveOddRateTraces
import matplotlib.pyplot as plt

# Script to scan through SEED files and extract feature values
archiveDir='/home/andrewr/Desktop/Python/Pickgot/MSEED/MM_Noise'
outDir='/home/andrewr/Desktop/Python/EventDetection_MM/FeatureArchive_Noise_hRes'
sliceLen=120.0 # How long a given section of a stream will be
sliceBuff=5.0 # How much buffer time on either side to trim after filtering (rid of edge effects)
bpFreqs=[0.5,49.9] # Bandpass frequencies
winLen,interLen=0.5,0.12 # Window and interval length for feature extraction
featureAvgLen=1.5 # Averaging length for features (time-averaging)
dopInterLen=0.02 # How often the DOP is initially calculated (average, spike, and difference are taken)
numBands=6 # Number of frequency bands to be extracted from spectrograms
# To save to numpy directory
sumFile='./MM_NoiseSummary_UTM11.csv' # Locations in X,Y,Z (down is positive)
staFile='./MM_stations_UTM11.csv' # Locations must be in km, going Y,X,Z (down is positive)
npyDir='./FeatureArchive_Noise_hRes_npy' # Save directory for when doing the regression

# Have to make the sliceLen a multiple of the interLen, so that different slices times can match up
sliceLen-=sliceLen%interLen

if not os.path.exists(outDir):
    os.makedirs(outDir)

# Extract only channels with the minimum trace duration, and which have all three components H1,H2,Z
# also, return as a sorted array so that the channels for a given station are grouped together (in above order)
def getTraces3C(stream,traceDurMin):
    # Before merging, see if each channel has the minimum duration
    staChas=np.array([aTrace.stats.station+'.'+aTrace.stats.channel for aTrace in stream])
    lenData=np.array([aTrace.stats.npts*aTrace.stats.delta for aTrace in stream])
    failLenStas=[]
    for aStaCha in np.unique(staChas):
        if np.sum(lenData[np.where(staChas==aStaCha)])<traceDurMin*0.99:
            print aStaCha,'not long enough',traceDurMin,np.sum(lenData[np.where(staChas==aStaCha)])
            failLenStas.append(aStaCha.split('.')[0])
    # Merge any overlapping data, and fill any gaps (gaps will fail via failLenStas)
    try:
        stream.merge(method=1,fill_value='interpolate')
    except:
        # Remone any traces which have odd sampling rates
        stream=RemoveOddRateTraces(stream)
        stream.merge(method=1,fill_value='interpolate')
    # Check to make sure it has three channels
    unqStas,staCounts=np.unique([aTrace.stats.station for aTrace in stream],return_counts=True)
    if np.max(staCounts)>3: print 'more than three channels present for '+str(unqStas[np.where(staCounts>3)])
    unqStas=list(unqStas)
    pass3CIdxs=[i for i in range(len(stream)) if staCounts[unqStas.index(stream[i].stats.station)]==3]
    # Add the completed 3C data to the output stream
    trimStream=EmptyStream()
    for i in range(len(stream)):
        if (stream[i].stats.station not in failLenStas) and (i in pass3CIdxs):
            trimStream+=stream[i]
    # Sort primarily by station, then by channel
    trimStream.sort(keys=['station','channel'])
    return trimStream

# Extract only 3C traces which are time aligned (within a sample point)
# ... this is done after traces are comfirmed to have all 3C (only) and are sorted by station
# ... for now just removing if not aligned
def alignTraces3C(stream):
    startTimes=np.array([aTrace.stats.starttime for aTrace in stream])
    npts=np.array([aTrace.stats.npts for aTrace in stream])
    passAlign=[]
    for i in range(0,len(stream),3):
        if np.max(startTimes[i:i+3]-np.min(startTimes[i:i+3]))>stream[i].stats.delta:
            print stream[i].stats.station,'data was not aligned perfectly'
            continue
        elif len(np.unique(npts[i:i+3]))!=1:
            print stream[i].stats.station,'had an inconsistent number of samples'
            continue
        passAlign+=range(i,i+3)
    # Get the traces which passed the test
    trimStream=EmptyStream()
    for i in range(len(stream)):
        if i in passAlign:
            trimStream+=stream[i]
    return trimStream
    
# Function to extract DOP values, and place them into a trace object
def getTracesDOP(stream,featureAvgLen,dopInterLen,winLen,interLen):
    outStream=EmptyStream()
    stats=TraceStats()
    stats.delta=interLen
    #For each trio of channels... 
    for i in range(0,len(stream),3):
        # ...set up this traces stats
        stats.network=stream[i].stats.network
        stats.station=stream[i].stats.station
        stats.location=stream[i].stats.location
        # ...get the dop values, and convert it to trace
        dopVals,dopTimes=getDOP(stream[i:i+3],featureAvgLen,dopInterLen)
        dopStats=getDOPStats(dopVals,dopTimes,winLen,interLen,dopInterLen)
        for aKey,aChaName in ['Avg','DP1'],['Change','DP2'],['Spike','DP3']:
            stats.channel=aChaName
            stats.starttime=dopStats['Times'][0]
            stats.npts=len(dopStats['Times'])
            outStream+=Trace(data=dopStats[aKey],header=stats)
    return outStream

def getTracesSono(stream,winLen,interLen,freqBands,freqNormIdxLen):
    outStream=EmptyStream()
    stats=TraceStats()
    stats.delta=interLen
    #For each trio of channels... 
    for i in range(0,len(stream),3):
        # Get the frequency content of different frequency bands averaged over a given time range...
        # ... and another normalized along the frequency band axis
        h1Times,h1SonoNormTime,h1SonoFreqDiff=getSonogram(stream[i],winLen,interLen,freqBands,freqNormIdxLen)
        h2Times,h2SonoNormTime,h2SonoFreqDiff=getSonogram(stream[i+1],winLen,interLen,freqBands,freqNormIdxLen)
        vertTimes,vertSonoNormTime,vertSonoFreqDiff=getSonogram(stream[i+2],winLen,interLen,freqBands,freqNormIdxLen)
        # Check first to see if the times all line up (sanity check)
        if np.sum(np.abs(h1Times-h2Times))>interLen*0.1 or np.sum(np.abs(h1Times-vertTimes))>interLen*0.1:
            print 'Times were misaligned in the sonogram extraction, how?'
            continue
#        ## DelMe ##
#        tr=stream[i]
#        plt.figure(figsize=(18,10))
#        plt.subplot(3,1,1)
#        plt.title(tr.stats.station)
#        plt.plot(tr.times()+tr.stats.starttime.timestamp,tr.data,'r')
#        plt.xlim(h1Times[0]-0.5*interLen,h1Times[-1]+0.5*interLen)
#        plt.subplot(3,1,2)
#        plt.imshow(h1SonoNormTime,interpolation='none',aspect='auto')
#        plt.show()
#        ## DelMe ##
        # Average the horizontal sonograms
        horiSonoNormTime=(h1SonoNormTime+h2SonoNormTime)/2.0
        horiSonoFreqDiff=(h1SonoFreqDiff+h2SonoFreqDiff)/2.0
        # Set up this traces stats
        stats.network=stream[i].stats.network
        stats.station=stream[i].stats.station
        stats.location=stream[i].stats.location
        stats.starttime=vertTimes[0]
        stats.npts=len(vertTimes)
        i=0
        for aSono in [vertSonoNormTime,horiSonoNormTime,vertSonoFreqDiff,horiSonoFreqDiff]:
            for aBand in aSono: 
                stats.channel='S'+str(i).zfill(2)
                outStream+=Trace(data=aBand,header=stats)
                i+=1
    return outStream

# To save these as numpy arrays
# Load data directly from the archive, and save as per-event files, formatted
def stream2npy(stream,numBands,aEveSumLine,staInfo,saveFolder):
    staDict=staInfo2Dict(staInfo)
    # Split up the event summary line to get ID, origin time and event location
    aID=aEveSumLine[0]
    aEveTime=UTCDateTime(aEveSumLine[5])
    aEveLoc=aEveSumLine[1:4].astype(float)
    # For each station...
    seenStas=np.unique([aTrace.stats.station for aTrace in stream])
    # ... start adding each stations observations
    for aSta in seenStas:
        staData=[]
        # Get the time array and hypocentral distance array
        staStream=stream.select(station=aSta)
        nPts=[aTrace.stats.npts for aTrace in staStream]
        maxNpts,minNpts=np.max(nPts),np.min(nPts)
        # Check first to see that the data has all of the features, and consitent time indicies...
        # ... it is possible that the DOP features has 1 less sample, as each set of data is...
        # ... collected separately with slightly different methods (ie. DOP vs Sono)
        if len(staStream)!=(4*numBands+3) or maxNpts-minNpts>1: 
            print aSta,'didnt have all features'
            continue
        times=staStream[0].times()+staStream[0].stats.starttime.timestamp-aEveTime.timestamp
        aStaLoc=staDict[aSta]['Loc']
        hypDist=np.sum((aStaLoc-aEveLoc)**2)**0.5
        hypArr=np.ones((len(times)))*hypDist
        # Sort by channel name
        staStream.sort(keys=['channel'])
        # Append all data
        staData.append(hypArr[:minNpts])
        staData.append(times[:minNpts])
        for aTrace in staStream:
            staData.append(aTrace.data[:minNpts])
        staData=np.array(staData,dtype=float)
        # Add this array to the saved directory
        np.save(saveFolder+'/'+aID.zfill(6)+'_'+aSta+'_FeatureData',staData)
    
def main():
    # Get the center frequency of the lowest band s.t. the minimum freq is the best possibly resolution (for sonograms)
    minBandFreq=1/winLen
    freqBands=float(minBandFreq)*(5.0/3)**np.arange(numBands+1)
    freqNormIdxLen=int(featureAvgLen/interLen) # How many indicies (lengths based on the interLen) ...
                                               # ...to average the energy of the frequency content over
    overLapLen=featureAvgLen+winLen # How much overlap there will be between slices, as time-averaging is required
    # Load in catalog summary and station locations (for later, when printing out as npy arrays)
    sumInfo=np.genfromtxt(sumFile,delimiter=',',dtype=str)
    staInfo=np.genfromtxt(staFile,delimiter=',',dtype=str)
    if not os.path.exists(npyDir):
        os.makedirs(npyDir)
    # For each MSEED file, filter and sort (primary station, secondary channel)
    for aFile in sorted(os.listdir(archiveDir)):
        print aFile
        stream=read(archiveDir+'/'+aFile)
        # Figure out the earliest time in this stream, and start getting data from there...
        # ... trim the first piece of stream to ensure that the data aligns (so no remainder using the interLen)
        firstTime=np.min([aTrace.stats.starttime for aTrace in stream])+sliceBuff+overLapLen
        firstTime=firstTime-(firstTime.timestamp%interLen)+interLen
        lastTime=np.max([aTrace.stats.endtime for aTrace in stream])-sliceBuff
        sliceStarts=np.arange(firstTime,lastTime,sliceLen)
        outName=aFile.replace('.seed','.pickle')
        outStream=EmptyStream()
        # Segment the MSEED into slices
        for i,t in enumerate(sliceStarts):
            # If on the second last slice, see if it should include just a bit more
            if (i==len(sliceStarts)-2 and sliceStarts[-1]!=lastTime) or len(sliceStarts)==1:
                thisSliceLen=lastTime-t
            # If we included the last bit of data from last loop, this doesn't have enough data anyhow
            elif i==len(sliceStarts)-1 and sliceStarts[-1]!=lastTime:
                break
            # Otherwise take the defined length
            else:
                thisSliceLen=sliceLen
            streamSlice=stream.slice(t-overLapLen-sliceBuff,t+thisSliceLen+sliceBuff)
            # Ensure that each trace has enough data points...
            # ...and has all three channels for a given station
            # ...this stream will be have a stations traces grouped
            streamSlice=getTraces3C(streamSlice,thisSliceLen+overLapLen+sliceBuff*2)
            # Ensure that the data is time aligned
            streamSlice=alignTraces3C(streamSlice)
            # Detrend and filter
            streamSlice.detrend('linear')
            streamSlice.filter(type='bandpass',freqmin=bpFreqs[0],freqmax=bpFreqs[1],corners=2,zerophase=True)
            # Convert to acceleration (for the DOP features)
            streamSliceAccel=streamSlice.copy().differentiate()
            # Trim off the edges of the stream...
            # ...removing an extra interLen at the end to be rid of any overlaps
            streamSlice.trim(t-overLapLen,t+thisSliceLen-interLen)
            streamSliceAccel.trim(t-overLapLen,t+thisSliceLen-interLen)
            # Start placing the features into traces
            outStream+=getTracesDOP(streamSliceAccel,featureAvgLen,dopInterLen,winLen,interLen)
            outStream+=getTracesSono(streamSlice,winLen,interLen,freqBands,freqNormIdxLen)
        outStream._cleanup()
        outStream.write(outDir+'/'+outName,format='PICKLE')
        # Also write this out as a numpy array
        aID=int(aFile.split('_')[0])
        wantArg=np.where(sumInfo[:,0].astype(int)==aID)[0][0]
        stream2npy(outStream,numBands,sumInfo[wantArg],staInfo,npyDir)

if __name__ == "__main__":
    main()