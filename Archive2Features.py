import numpy as np
from obspy import read, UTCDateTime
from obspy import Stream as EmptyStream
import os
from Archive2Features_ForCatalog import getTraces3C,alignTraces3C,getTracesDOP,getTracesSono
import matplotlib.pyplot as plt

# Script to scan through SEED files and extract feature values
archiveDir='/home/andrewr/Desktop/Python/Pickgot/MSEED/MM_Archive'
outDir='/home/andrewr/Desktop/Python/EventDetection_MM/FeatureArchive_hRes'
sliceLen=120.0 # How long a given section of a stream will be
sliceBuff=5.0 # How much buffer time on either side to trim after filtering (rid of edge effects)
bpFreqs=[0.5,49.9] # Bandpass frequencies
winLen,interLen=0.5,0.12 # Window and interval length for feature extraction
featureAvgLen=1.5 # Averaging length for features (time-averaging)
dopInterLen=0.02 # How often the DOP is initially calculated (average, spike, and difference are taken)
numBands=6 # Number of frequency bands to be extracted from spectrograms

# Have to make the sliceLen a multiple of the interLen, so that different slices times can match up
sliceLen-=sliceLen%interLen

if not os.path.exists(outDir):
    os.makedirs(outDir)
    
def main():
    # Get the center frequency of the lowest band s.t. the minimum freq is the best possibly resolution (for sonograms)
    minBandFreq=1/winLen
    freqBands=float(minBandFreq)*(5.0/3)**np.arange(numBands+1)
    freqNormIdxLen=int(featureAvgLen/interLen) # How many indicies (lengths based on the interLen) ...
                                               # ...to average the energy of the frequency content over
#    print freqBands
#    quit()
    overLapLen=featureAvgLen+winLen # How much overlap there will be between slices, as time-averaging is required
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
        outName=UTCDateTime(firstTime-interLen).strftime('%Y%m%d.%H%M%S.%f.pickle')
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
        
if __name__ == "__main__":
    main()