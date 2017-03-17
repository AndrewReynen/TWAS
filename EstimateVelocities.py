import sys,os
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from obspy import UTCDateTime
sys.path.insert(0, '/home/andrewr/Desktop/Python')
from NOC_Functions import GetInfoFromSoln,Reproject

pickFolder='/home/andrewr/Desktop/Python/Pickgot/WantedPicks/MM_NCEDC'
staFile='./MM_stations_UTM11.csv'
sumFile='./NCEDC_Trimmed_UTM11.csv'

staInfo=np.genfromtxt(staFile,delimiter=',',dtype=str)
sumInfo=np.genfromtxt(sumFile,delimiter=',',dtype=str)

# Turn this into a dictionary of Dict[StaName]=StaLoc
staLoc={}
for aEntry in staInfo:
    staLoc[aEntry[0]]=aEntry[np.array([2,1,3])].astype(float)

# Turn events into dictionary of Dict[EveID]=EveLoc,EveTime
eveLoc={}
for aEntry in sumInfo:
    aID=str(int(aEntry[0]))
    eveLoc[aID]=aEntry[np.array([1,2,3,5])]
    eveLoc[aID][3]=UTCDateTime(eveLoc[aID][3]).timestamp
    eveLoc[aID]=eveLoc[aID].astype(float)

# Read in the picks, and figure out its travel time
obs={'P':[],'S':[]} # Will contain trios of travel time, epicentral, and hypocentral distance
for aFile in os.listdir(pickFolder):
    picks=GetInfoFromSoln(pickFolder,aFile)[0]
    aEveLoc=eveLoc[str(int(aFile.split('_')[0]))]
    for aPick in picks:
        tt=aPick.Time.timestamp-aEveLoc[3]
        aStaLoc=staLoc[aPick.Sta]
        hypDist=np.sum((aStaLoc-aEveLoc[:3])**2)**0.5
        epiDist=np.sum((aStaLoc[:2]-aEveLoc[:2])**2)**0.5
        obs[aPick.Type].append([tt,epiDist,hypDist,aEveLoc[2]])
obs['P']=np.array(obs['P'],dtype=float)
obs['S']=np.array(obs['S'],dtype=float)

velInfo={}
## Perform least squares
for aType in ['P','S']:
    X=np.matrix(np.vstack((obs[aType][:,0],np.ones(len(obs[aType])))).T)
    Y=np.matrix(obs[aType][:,2].reshape((len(obs[aType]),1)))
    b=np.linalg.inv(X.T*X)*(X.T*Y)
    velInfo['V'+aType]=round(float(b[0]),2)
    velInfo['D'+aType]=round(float(b[1]/b[0]),2)


## Try fitting myself
#velInfo['VP']=4.7
#velInfo['DP']=0.2
#velInfo['VS']=2.7
#velInfo['DS']=0.2

# Plot some curves from given 1D velocity model (Hyp,Epi,TT)
#dep32=np.genfromtxt('TT_3.2.csv',delimiter=',',dtype=float)
#dep50=np.genfromtxt('TT_5.0.csv',delimiter=',',dtype=float)
#plt.plot(dep32[:,0],dep32[:,2],'g',lw=3)
#plt.plot(dep32[:,0],dep32[:,2]*1.75,'y',lw=3)
#plt.plot(dep50[:,0],dep50[:,2],'r')
#plt.plot(dep50[:,0],dep50[:,2]*1.75,'b')

print velInfo

dists=np.linspace(0,8,2)
#plt.scatter(obs['P'][:,2],obs['P'][:,0],c=obs['P'][:,3],s=30)
#plt.scatter(obs['S'][:,2],obs['S'][:,0],c=obs['S'][:,3],s=30)
plt.plot(obs['P'][:,2],obs['P'][:,0],'go')
plt.plot(obs['S'][:,2],obs['S'][:,0],'yo')
plt.plot(dists,(dists-velInfo['VP']*velInfo['DP'])/velInfo['VP'],'g')
plt.plot(dists,(dists-velInfo['VS']*velInfo['DS'])/velInfo['VS'],'y')
# Give title (velocities and delays), and axis labels
keys=['VP','DP','VS','DS']
titleString='' 
for aKey in keys:
    titleString+=aKey+':'+str(velInfo[aKey])+' '
plt.title(titleString)
plt.xlabel('Hypocentral Distance (km)')
plt.ylabel('Travel Time (sec)')
plt.xlim(0,10)
plt.ylim(0,3.5)
plt.show()
