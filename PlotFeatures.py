import matplotlib.pyplot as plt
from obspy import UTCDateTime,read
import numpy as np
import os,sys
sys.path.insert(0, '/home/andrewr/Desktop/Python')
from Reynen_Functions import downSampleData,fillGridData

npyFolder='./FeatureArchive_NCEDC_npy'
catSumFile='./NCEDC_Trimmed_UTM11.csv' # Locations in X,Y,Z (down is positive)
staFile='./MM_stations_UTM11.csv' # Locations must be in km, going Y,X,Z (down is positive)
outName='Feature_vs_XT.png'
gridXInfo,gridYInfo=[2,9,0.2],[0,5,0.25]
nf=35
# Velocity and delays just for plotting vs. the reference
Vp,Vs=4.7,2.7
Dp,Ds=0.2,0.2

timesLen=int((gridYInfo[1]-gridYInfo[0])/float(gridYInfo[2]))

sumInfo=np.genfromtxt(catSumFile,delimiter=',',dtype=str)
staInfo=np.genfromtxt(staFile,delimiter=',',dtype=str)
nBand=(nf-3)/4

def PSlines():
    d=np.linspace(gridXInfo[0],gridXInfo[1],2)
    tp=(d-Vp*Dp)/Vp
    ts=(d-Vs*Ds)/Vs
    return d,tp,ts
        
def main():
    # Read in the data, note that features are row-wise
    data=np.empty((nf+2,0))
    ## Much better ways to load then this - maybe just append then reshape at end?##
    for aFile in sorted(os.listdir(npyFolder)):
        fileData=np.load(npyFolder+'/'+aFile)
        # Trim the data to be within the wanted ranges
        wantCols=np.where((fileData[0]>gridXInfo[0])&(fileData[0]<gridXInfo[1])&
                          (fileData[1]>gridYInfo[0])&(fileData[1]<gridYInfo[1]))[0]
        if len(wantCols)>0:
            data=np.hstack((data,fileData[:,wantCols]))
    print data.shape
    # Down sample the data
    xData,yData,zData=downSampleData(data,0,1,range(2,nf+2),gridXInfo,gridYInfo,[np.mean])                
    # Plot also the predicted lines for P and S arrivals
    d,tp,ts=PSlines()
    for aRange,aOrientation,vmin,vmax in [[range(3),'DOP',-999,999],
                                          [range(3,3+nBand),'Hori_NormTime',0,0.4],
                                          [range(3+nBand,3+2*nBand),'Vert_NormTime',0,0.4],
                                          [range(3+2*nBand,3+3*nBand),'Hori_FreqDiff',-0.15,0.15],
                                          [range(3+3*nBand,3+4*nBand),'Vert_FreqDiff',-0.15,0.15]]:
        fig=plt.figure(figsize=(12,12))
        fig.suptitle(aOrientation)
        for aBandID in aRange:
            # For each DOP give different ranges for coloring
            if aBandID==0:
                vmin,vmax=0,1
            elif aBandID==1:
                vmin,vmax=-0.1,0.1
            elif aBandID==2:
                vmin,vmax=0,0.35
            zGrid=fillGridData(xData,yData,zData[aBandID][0])[-1]
            ax=fig.add_subplot(3,3,aBandID%nBand+1)
            ax.set_title('Band:'+str(aBandID%nBand))
            ax.imshow(zGrid,origin='lower',aspect='auto',cmap=plt.get_cmap('jet'),interpolation='none',
#                             extent=[gridXInfo[0],gridXInfo[1],gridYInfo[0],gridYInfo[1]],
                             extent=[np.min(data[0]),np.max(data[0]),np.min(data[1]),np.max(data[1])],
                             vmin=vmin,vmax=vmax)
            ax.plot(d,tp,'k')
            ax.plot(d,ts,'k')
            ax.set_ylim(gridYInfo[0],gridYInfo[1])
            ax.set_xlim(gridXInfo[0],gridXInfo[1])
        # Save the plot
        fig.savefig(aOrientation+'_'+outName, bbox_inches = 'tight',pad_inches = 0.2)
        plt.close()
main()
