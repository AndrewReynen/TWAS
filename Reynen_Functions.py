from matplotlib.widgets import Lasso
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection
from matplotlib import path
from scipy import signal
from obspy import read, UTCDateTime
from obspy import Stream as EmptyStream
from sklearn.decomposition import PCA
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import os

class LassoArray(object):
    # Example usage:
    # wantIdxs=LassoArray(data,0,1).wantIdxs
    def __init__(self, data,xCol,yCol):
        # set up the plot
        self.axes = plt.axes()
        self.canvas = self.axes.figure.canvas
        self.colorIn=colorConverter.to_rgba('red')
        self.colorOut=colorConverter.to_rgba('blue')
        self.data = data
        self.xys = [(float(d[xCol]), float(d[yCol])) for d in data]
        self.wantIdxs=np.array([])
        facecolors = [self.colorOut for anEntry in data]
        fig = self.axes.figure
        self.collection = RegularPolyCollection(
            fig.dpi, 6, sizes=(50,),
            facecolors=facecolors,
            offsets=self.xys,
            transOffset=self.axes.transData)
        self.axes.add_collection(self.collection)
        # enable extra interactions
        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        # set some nice limits
        xmin,xmax=np.min(data[:,xCol].astype(float)),np.max(data[:,xCol].astype(float))
        ymin,ymax=np.min(data[:,yCol].astype(float)),np.max(data[:,yCol].astype(float))
        plt.xlim(xmin-0.05*(xmax-xmin),xmax+0.05*(xmax-xmin))
        plt.ylim(ymin-0.05*(ymax-ymin),ymax+0.05*(ymax-ymin))
        plt.show()
    
    # Manage when the lasso line has been finished
    def callback(self, verts):
        facecolors = self.collection.get_facecolors()
        p = path.Path(verts)
        ind = p.contains_points(self.xys)
        for i in range(len(self.xys)):
            if ind[i]:
                facecolors[i] = self.colorIn
            else:
                facecolors[i] = self.colorOut
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        self.wantIdxs=np.array([i for i in range(len(facecolors)) if 
                                np.sum(np.abs(np.array(facecolors[i])-np.array(self.colorIn)))==0])
        del self.lasso        
    
    # Manage when the lasso line starts to draw
    def onpress(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

class Pick(object):
    def __init__(self,PickTime,PickType,Net,Sta,Cha):
        self.Time=PickTime
        self.Type=PickType
        self.Net=Net
        self.Sta=Sta
        self.Cha=Cha

# Read in picks from a pick file
def readPicks(SolnFileFolder,SolnFileName):
    Picks=[]
    with open(SolnFileFolder+'/'+SolnFileName) as f:
        Info = [x.strip('\n') for x in f.readlines()]
    for i in range(0,len(Info)):
        aLine=Info[i].split(',')
        if Info[i][0:6]=='phase,':
            PickTime=UTCDateTime(aLine[6])
            PickType=aLine[5][0]
            Net=aLine[1]
            Sta=aLine[2]
            Cha=aLine[3]
            Picks.append(Pick(PickTime,PickType,Net,Sta,Cha))
    return Picks

# Write a pick file (no event information)
def writePicks(PicksFolder,FileName,PickSet):
    out_file=open(PicksFolder+'/'+FileName+'.aeq','w')
    for aPick in PickSet:
        aPickTime=(aPick.Time).strftime('%Y-%m-%d %H:%M:%S.%f')
        out_file.write("phase,"+aPick.Net+","+aPick.Sta+","+aPick.Cha+",,"+aPick.Type+","+aPickTime+",,,0,,,\n")
    out_file.close()
    
# Convert picks from aeq to lazylyst format...
# ...assume file name using same convention
def aeq2lazylystPicks(aeqDir,picksDir):
    # Make the new directory if it doesn't exist
    if not os.path.exists(picksDir):
        os.makedirs(picksDir)
    # Loop through all files, get picks and convert to an array if [sta,pickType,timestamp(s)]
    for aFile in os.listdir(aeqDir):
        ext='.'+aFile.split('.')[-1]
        picks=GetInfoFromSoln(aeqDir,aFile)[0]
        pickArr=[[aPick.Sta,aPick.Type,Decimal(aPick.Time.timestamp)] for aPick in picks]
        pickArr=np.array(pickArr,dtype=str)
        np.savetxt(picksDir+'/'+aFile.replace(ext,'.picks'),pickArr,fmt='%s',delimiter=',')

# Read from a pick file
def GetInfoFromSoln(SolnFileFolder,SolnFileName):
    Picks=[]
    Lon,Lat,Dep,Mag,EveTime='nan','nan','nan','nan','nan'
    with open(SolnFileFolder+'/'+SolnFileName) as f:
        Info = [x.strip('\n') for x in f.readlines()]
    for i in range(0,len(Info)):
        aLine=Info[i].split(',')
        if Info[i][0:7]=='origin,':
            EveTime=UTCDateTime(aLine[1])
            Lat=aLine[4]
            Lon=aLine[5]
            Dep=aLine[6]
        if Info[i][0:6]=='phase,':
            PickTime=UTCDateTime(aLine[6])
            PickType=aLine[5][0]
            Net=aLine[1]
            Sta=aLine[2]
            Cha=aLine[3]
            Picks.append(Pick(PickTime,PickType,Net,Sta,Cha))
        if Info[i][0:10]=='magnitude,':
            Mag=aLine[3]
    return Picks,Lon,Lat,Dep,Mag,EveTime
    
#aeq2lazylystPicks('/home/andrewr/Desktop/Python/EventDetection_MM/NewDetectionPicks/2016Dec16',
#                  '/home/andrewr/Desktop/Python/EventDetection_MM/NewDetectionPicks/2016Dec16Lazy')
                  
# Return four arrays, MatchedMain,UnmatchedMain,MatchedOff,UnmatchedOff
# based off the best matching events, within a leniance (of the event times, in seconds)
def Match1to1Summary(Main,Off,Leniance):
    NumNewMatches=1
    MatchedMain,MatchedOff=[],[]
    # As we could have a main event closer to two off events, iterate until all
    # closest matches within the leniance window are found
    while NumNewMatches!=0 and len(Main)!=0 and len(Off)!=0:
        TimesMain=np.array([UTCDateTime(aRow[5]).timestamp for aRow in Main])
        TimesOff=np.array([UTCDateTime(aRow[5]).timestamp for aRow in Off])
        # Make a 2D array of all potential time differences
        # Using the absolute differences
        Diffs=np.abs(TimesOff.reshape(1,-1)-TimesMain.reshape(-1,1))
        # Figure out which index of the main array best matched a value of the Off array
        IdxMains=Diffs.argmin(axis=0) # Get closest index of each row
        Resids=np.diagonal(Diffs[IdxMains,])  # Resort the differences, by above indicies, and grab absolute residuals
        # Sort these arrays, so that the closest match is seen first
        ResidSort=np.argsort(Resids)
        IdxMains,Resids,IdxOffs=IdxMains[ResidSort],Resids[ResidSort],np.arange(len(Resids))[ResidSort]
        # Find out how many new matches there were
        NumNewMatches=0
        SeenMain,SeenOff=[],[] # Seen indicies for this iteration
        for IdxMain,IdxOff,aResid in zip(IdxMains,IdxOffs,Resids):
            # If over the residual threshold, skip!
            if aResid>Leniance:
                continue
            # Don't do any double-matching of events...
            if IdxOff in SeenOff or IdxMain in SeenMain:
                continue
            # Send these entries to the matched!
            MatchedMain.append(Main[IdxMain])
            MatchedOff.append(Off[IdxOff])
            SeenMain.append(IdxMain)
            SeenOff.append(IdxOff)
            NumNewMatches+=1
        # Reset the Main and Off arrays (remaining entries)
        Main=[Main[i] for i in range(len(Main)) if i not in SeenMain]  
        Off=[Off[i] for i in range(len(Off)) if i not in SeenOff]
    return np.array(MatchedMain),np.array(Main),np.array(MatchedOff),np.array(Off)
    
# Convert between projections, must make sure input units are right
def Reproject(XYs,Xcol,Ycol,EPSG1,EPSG2):
    XYs=np.array(XYs,dtype=str) # Convert to string to ensure nothing is lost
    Proj1=pyproj.Proj(init='epsg:'+str(EPSG1))
    Proj2=pyproj.Proj(init='epsg:'+str(EPSG2))
    for i in range(len(XYs)):
        aX,aY=pyproj.transform(Proj1,Proj2,float(XYs[i][Xcol]),float(XYs[i][Ycol]))
        XYs[i][Xcol],XYs[i][Ycol]=aX,aY
    return XYs

# Given a single stations data on all three channels...
# ...return the dop for the end of the window time, with specified intervals
# ...!assume that the passed data is time aligned!
def getDOP(staStream3comp,windowLen,intervalLen):
    dopVals,dopTimes=[],[]
    # get the data, and delta
    data=np.array([aTrace.data for aTrace in staStream3comp])
    delta=staStream3comp[0].stats.delta
    times=staStream3comp[0].times()+staStream3comp[0].stats.starttime.timestamp
    windowIdxLen=int(np.ceil(windowLen/delta))
    intervalIdxLen=float(intervalLen/delta) #leaving as float, may have to round
    if len(np.unique([aTrace.stats.station for aTrace in staStream3comp]))!=1 or len(staStream3comp)!=3:
        print 'got a non-unique set of stations, something went wrong'
        quit()
    for i in np.arange(0,len(data[0])-windowIdxLen,intervalIdxLen):
        i=int(round(i))
        if i>=len(data[0])-windowIdxLen:
            continue
        eVals = np.linalg.svd(np.cov(data[:,i:i+windowIdxLen]),compute_uv=0)
        dop=0.5*((eVals[0]-eVals[1])**2+(eVals[0]-eVals[2])**2+(eVals[1]-eVals[2])**2)/(np.sum(eVals**2))
        dopVals.append(dop)
        dopTimes.append(times[i+windowIdxLen])
    return np.array(dopVals),np.array(dopTimes)

# Get down-sampled wanted stats from a DOP time series...
# ...values reported from the center of a given window
# ...must ensure that the original sampling rate << intervalLen given here
def getDOPStats(dopVals,dopTimes,windowLen,intervalLen,dopInterLen):
    # set up output dictionary
    dopStats={'Spike':[],'Avg':[],'Change':[],'Times':[]} 
    # if there is only one entry or less return nothing    
    if len(dopTimes)<2:
        return dopStats
    windowIdxLen=int(windowLen/(dopInterLen))
    intervalIdxLen=float(intervalLen)/(dopInterLen) #leaving as float, may have to round
    # start grabbing the wanted values
    lastAvg='' # placeholder
    for i in np.arange(0,len(dopTimes)-windowIdxLen,intervalIdxLen):
        i=int(round(i))
        dopFrag=dopVals[i:i+windowIdxLen]
        # take the largest spike in the dop fragment
        dopStats['Spike'].append(np.max(np.abs(dopFrag[1:]-dopFrag[:-1])))
        dopStats['Avg'].append(np.average(dopFrag))
        # report zero change on the first sample
        if i==0:
            dopStats['Change'].append(0.0)
        else:
            dopStats['Change'].append(dopStats['Avg'][-1]-lastAvg)
        # down sampled values are reported at the center of the window
        dopStats['Times'].append(dopTimes[i+windowIdxLen/2])
        lastAvg=dopStats['Avg'][-1]
    # convert the lists to numpy arrays
    for aKey in dopStats.keys():
        dopStats[aKey]=np.array(dopStats[aKey])
    return dopStats 

# Extract a sonogram from a given trace
def getSonogram(trace,winLen,interLen,freqBands,freqNormIdxLen):
    numBands=len(freqBands)-1
    sampleRate=trace.stats.sampling_rate
    f, t, psd=signal.spectrogram(trace.data, trace.stats.sampling_rate,
                                 nperseg=int(sampleRate*winLen),noverlap=int(sampleRate*(winLen-interLen)))
    # Reference the times to the start of the trace
    t+=trace.stats.starttime.timestamp
    rowIdxs=np.digitize(f,freqBands)-1
    sonogram=np.zeros((numBands,len(t)))
    sonoTimeNorm=np.zeros((numBands,len(t)))
    sonoFreqNorm=np.zeros((numBands,len(t)))
    # Over all wanted frequency bands... 
    for i in range(numBands):
        # ...sum the content
        wantRows=np.where(rowIdxs==i)[0]
        sonogram[i]=np.sum(psd[wantRows],axis=0)
        # ...make a sonogram which is normalized in windows over every few samples (time axis)
        mvSum=np.cumsum(sonogram[i])
        mvSum[freqNormIdxLen:] = mvSum[freqNormIdxLen:] - mvSum[:-freqNormIdxLen]
        sonoTimeNorm[i]=sonogram[i]/mvSum
    # Make a sonogram which is normalized via the frequencies at a given time (frequency band axis)...
    sonoFreqNorm=sonogram/(np.sum(sonogram,axis=0))
    # ...calculate the moving average across all bands
    mvSum=np.cumsum(sonoFreqNorm,axis=1)
    mvSum[:,freqNormIdxLen:] = mvSum[:,freqNormIdxLen:] - mvSum[:,:-freqNormIdxLen]
    # ...get the average value of each band over thes given time samples
    mvAvg=mvSum[:,freqNormIdxLen-1:-1]/freqNormIdxLen
    # ...calculate the difference between the average and the next sample
    sonoFreqNorm=sonoFreqNorm[:,freqNormIdxLen:]
    sonoFreqDiff=sonoFreqNorm-mvAvg
    # Trim the first few samples, as they did not sum over enough values...
    # ...plus one extra to start at the same time as the FreqDiff (already trimmed) array
    sonoTimeNorm=sonoTimeNorm[:,freqNormIdxLen:]
    t=t[freqNormIdxLen:]
    return t,sonoTimeNorm,sonoFreqDiff

# Convert station file into a dictionary
def staInfo2Dict(staInfo):
    staDict={}
    for aEntry in staInfo:
        xyz=np.array([aEntry[2],aEntry[1],float(aEntry[3])],dtype=float)
        staDict[aEntry[0]]={'Loc':xyz}
    return staDict
    
# Compute given function over a 2D grid, with a unrolled array as input
# ...note that data given is given as rows, and gridInfo=[minVal,maxVal,spacing]
# ...return values are set into an array with [rowIdx][functionIdx]
def downSampleData(data,xRow,yRow,zRows,xGridInfo,yGridInfo,functions):
    xGridMin,xGridMax,dX=np.array(xGridInfo,dtype=float)
    yGridMin,yGridMax,dY=np.array(yGridInfo,dtype=float)
    x=data[xRow]
    y=data[yRow]
    # For the grid values...
    xi=np.arange(xGridMin,xGridMax+dX/1000.0,dX)
    yi=np.arange(yGridMin,yGridMax+dY/1000.0,dY)
    xgrid, ygrid = np.meshgrid(xi, yi)
    # Figure out which indicies of the input array go to a given grid cell
    IdxX=np.array(np.digitize(x,xi),dtype=str)
    IdxY=np.array(np.digitize(y,yi),dtype=str)
    # Make a unique set of indicie pairs (so that all initial data indicies can be grouped to these)
    IdxXY=np.array([IdxX[i]+'-'+IdxY[i] for i in range(len(IdxX))],dtype=str)
    UnqVals,Inverse,Counts=np.unique(IdxXY,return_inverse=True,return_counts=True)
    # Split the Unique values back into IdxX, and IdxY
    IdxX=np.array([Entry.split('-')[0] for Entry in UnqVals]).astype(int)
    IdxY=np.array([Entry.split('-')[1] for Entry in UnqVals]).astype(int)
    # Convert back to inputs units (from index values)
    wantX=xi[IdxX]-0.5*dX
    wantY=yi[IdxY]-0.5*dY
    IdxZ=np.split(np.argsort(Inverse), np.cumsum(Counts[:-1]))
    # for each parameter to be checked...
    wantZs=[]
    for i,zRow in enumerate(zRows):
        thisZRowVals=[]
        z=data[zRow]
        for aFunction in functions:
            thisZRowVals.append(np.array([aFunction(z[IdxZ[i]]) for i in range(len(UnqVals))]))
        wantZs.append(thisZRowVals)
    return wantX,wantY,np.array(wantZs,dtype=float)
    
# Given a set of data sampled in a grid, fill any missing values with nan
def fillGridData(xData,yData,zData,emptyAsNaN=False):
    # Figure out the minimum grid spacing, and grid size
    minX,maxX=np.nanmin(xData),np.nanmax(xData)
    minY,maxY=np.nanmin(yData),np.nanmax(yData)
    dX=np.min(np.abs(np.diff(np.sort(np.unique(xData)))))
    dY=np.min(np.abs(np.diff(np.sort(np.unique(yData)))))
    # Form the grids
    x=np.arange(minX,maxX+dX/1000.0,dX)
    y=np.arange(minY,maxY+dY/1000.0,dY)
    xGrid,yGrid=np.meshgrid(x,y)
    zGrid=np.empty(xGrid.shape)
    if emptyAsNaN:
        zGrid[:]=np.nan
    # For any point which did exist, fill it in (allowing for a bit of rounding error)
    for j,yVal in enumerate(yGrid[:,0]):
        for i,xVal in enumerate(xGrid[0]):
            checkArr=(np.abs(xData-xVal)<0.01*dX)&(np.abs(yData-yVal)<0.01*dY)
            if np.sum(checkArr)==1:
                zGrid[j][i]=zData[np.where(checkArr)]
    return xGrid,yGrid,zGrid

# Extract data from a feature archive, similar format as getting from an FDSN client (ex. IRIS)
# This assumes ONE sampling rate per station
def extractFeatureData(archiveDir,t1,t2,wantedStaChas,archiveFileLen=1800,fillVal=None):
    # If t1>t2, return nothing
    if t1>t2:
        print 't1 > t2'
        return EmptyStream()
    # If both times given as UTCDateTime objects, convert
    try:
        t1=t1.timestamp
        t2=t2.timestamp
    except:
        pass
    # Make a list of event times from the folder, for easier searching later
    fileTimes,files=[],os.listdir(archiveDir)
    for aFile in files:
        try:
            aFileTimeString=aFile.split('_')[-1].replace('.pickle','').replace('.seed','')
            fileTimes.append((UTCDateTime().strptime(aFileTimeString,'%Y%m%d.%H%M%S.%f')).timestamp)
        except:
            print aFileTimeString+' does not match the format %Y%m%d.%H%M%S.%f.{pickle/seed}'
            continue
    fileTimes,files=np.array(fileTimes,dtype=float),np.array(files,dtype=str)
    argSort=np.argsort(fileTimes)
    fileTimes,files=fileTimes[argSort],files[argSort]
    # Read the last file to see how long it goes on for
    lastTime=fileTimes[-1]+archiveFileLen
    # Catch the case where this where the asked time range is completely outside the archive data availability
    if t1>lastTime or t2<fileTimes[0]:
        print 'Archive contains data between '+str(UTCDateTime(fileTimes[0]))+' and '+str(UTCDateTime(lastTime)) 
        return EmptyStream()
    # Figure out what set of files are wanted
    firstIdx,secondIdx=np.interp(np.array([t1,t2]),fileTimes,np.arange(len(fileTimes)),
                                 left=0,right=len(fileTimes)-1).astype(int)
    stream=EmptyStream()
    # Read in all of the information
    for aFile in files[firstIdx:secondIdx+1]:
        aStream=read(archiveDir+'/'+aFile)
        for aSta,aCha in wantedStaChas:
            stream+=aStream.select(station=aSta,channel=aCha)
    try:
        # Merge traces which are adjacent
        stream.merge(method=1,fill_value=fillVal)
    except:
        stream=RemoveOddRateTraces(stream)
        # Merge traces which are adjacent
        stream.merge(method=1,fill_value=fillVal)
    # Trim to wanted times
    stream.trim(UTCDateTime(t1),UTCDateTime(t2))
    return stream
    
def RemoveOddRateTraces(stream):
    # If there are two traces with same ID, but different sampling rates...
    # ...take the one with more common sampling rate
    unqStaRates=[] # Array to hold rates, per station
    rates,stas=[],[]
    for tr in stream:
        if tr.stats.station+'.'+str(tr.stats.delta) not in unqStaRates:
            unqStaRates.append(tr.stats.station+'.'+str(tr.stats.delta))
            rates.append(tr.stats.delta)
            stas.append(tr.stats.station)
    # ...figure out which station has two rates 
    unqRate,countRate=np.unique(rates,return_counts=True)
    if len(unqRate)==1:
        print 'merge will fail, not issue of multiple sampling rates on same station'
    else:
        # ... and remove the traces with less common rates
        unqSta,countSta=np.unique(stas,return_counts=True)
        rmStas=unqSta[np.where(countSta!=1)]
        rmRates=unqRate[np.where(unqRate!=(unqRate[np.argmax(countRate)]))]
        trimRateStream=EmptyStream()
        for tr in stream:
            if tr.stats.station in rmStas and tr.stats.delta in rmRates:
                continue
            trimRateStream+=tr
        print 'stations:',str(rmStas),'had some traces removed (duplicate rates same channel)'
        stream=trimRateStream
    return stream
    
# Get the pca method, and how many of the vectors should be used...
# ...given an input set of data and how much variance should be covered
def getPCA(data,portionVar):
    data=np.array(data)
    # Remove the mean (allows PCA to work)...
    # ...and divide the standard deviation (ensure that a feature with larger numbers dominate the variance)
    dataStdev=np.std(data,axis=0)
    dataMean=np.mean(data,axis=0)
    data=(data-dataMean)/dataStdev
    pca = PCA()
    pca.fit(data)
    vecVarianceContrib=pca.explained_variance_ratio_
    vecNums=range(1,len(data[0])+1)
    numReqParams=int(np.ceil(np.interp([portionVar], np.cumsum(vecVarianceContrib),vecNums)[0]))
    return pca,numReqParams