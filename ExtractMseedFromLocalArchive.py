from obspy import UTCDateTime
import numpy as np
import os,sys
sys.path.insert(0, '/home/andrewr/Desktop/Python')
from Reynen_Functions import extractFeatureData

sumFile='OldRelocShelly_UTM11.csv'
outMseedDir='/home/andrewr/Desktop/Python/Pickgot/MSEED/MM_ShelOld'
archiveDir='/home/andrewr/Desktop/Python/Pickgot/MSEED/MM_Archive'
timeRange=[-5,15]
wantedStaChas=[['*','*']]
archiveFileLen=3600 # How long one archive file is
# If wanted to make an empty set of pick files
makePickFiles=True
emptyPickDir='/home/andrewr/Desktop/Python/Pickgot/Picks/MM_ShelOld'

if not os.path.exists(outMseedDir):
    os.makedirs(outMseedDir)
if not os.path.exists(emptyPickDir) and makePickFiles:
    os.makedirs(emptyPickDir) 

info=np.genfromtxt(sumFile,delimiter=',',dtype=str)
if len(info)==0:
    quit()
elif len(info.shape)==1:
    info=np.array([info],dtype=str)
for aEntry in info:
    eveTime=UTCDateTime(aEntry[5])
    outName=str(aEntry[0]).zfill(6)+'_'+eveTime.strftime('%Y%m%d.%H%M%S.%f')
    stream=extractFeatureData(archiveDir,eveTime+timeRange[0],eveTime+timeRange[1],
                              wantedStaChas,archiveFileLen=archiveFileLen,fillVal=None)
    for tr in stream:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled(np.mean(tr.data))
    print outName
    stream.write(outMseedDir+'/'+outName+'.seed',format='MSEED')
    if makePickFiles:
        outPickFile=open(emptyPickDir+'/'+outName+'.aeq','w')
        outPickFile.close()

