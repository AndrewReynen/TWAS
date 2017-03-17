from obspy import UTCDateTime
import sys,shutil
sys.path.insert(0, '/home/andrewr/Desktop/Python')
from Reynen_Functions import downSampleData,fillGridData,Match1to1Summary
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

#sumDir='./NewDetectionSummaries/2016Dec16'
#outName='./Merged_Summary_2016Dec16.csv'
#mergeSum=[]
## Loop through all files, and collect summaries
#for aFile in sorted(os.listdir(sumDir)):
#    if 'Summary.csv' not in aFile:
#        continue
#    evePack=np.genfromtxt(sumDir+'/'+aFile,delimiter=',',dtype=str)
#    if len(evePack)==0:
#        continue
#    if len(evePack.shape)==1:
#        evePack=np.array([evePack],dtype=str)
#    for entry in evePack:
#        mergeSum.append(entry)
#print len(mergeSum),'events in the merged summary'
#np.savetxt(outName,np.array(mergeSum),fmt='%s',delimiter=',',
#           header='ID,X,Y,X,Mag,DateTime,EveSumProb,NumSta')
        
#
#noiseIDs=[18,70,90,96,102,106,965,983,1390,1447,1545,1586,1609,1622,1625,1636,1700,1717,1722,1752,1759,
#          1769,1794,1816,1821,2364,2376,2502,2537,2545,2548,2555,2588,2591,2604,2036,2044,1908,1941,
#          1942,1951,1952,1956,1958,1959,2057,1968,2084,2158,2320,1982,2331,1996,2337,2347,2733,2755,
#          2760,2766,2775,2784,2788,2791,2795,2799,2801,2802,2804,2806,2811,2813,2815,2816,2818,2819]
#unsureIDs=[37,64,66,79,80,83,887,1317,1419,1488,1620,1866,1901,2136,2164,2200,2628,2692,2764,
#           1436,1544,1575,1963,1969,1972,1986,2027,2043,2300,2334,2406,2409,2549]
#farIDs=[86,89,2172,2169,2531,2577,2718]
#dupIDs=[2155,2180,2575,2581,2707,2769]
#
# Compare summaries
reySum=np.genfromtxt('./Merged_Summary_2016Dec16.csv',delimiter=',',dtype=str)
sum1=np.genfromtxt('./NewShelly_UTM11.csv',delimiter=',',dtype=str)
sum2=np.genfromtxt('./OldRelocShelly_UTM11.csv',delimiter=',',dtype=str)
staInfo=np.genfromtxt('./MM_stations_UTM11.csv',delimiter=',',dtype=str)

shelSum=np.vstack((sum1,sum2))

# Match them up
mm,um,mo,uo=Match1to1Summary(reySum,sum2,3.0)
print len(mm),len(um),len(mo),len(uo)

#
#reySum=reySum[np.where(reySum[:,6].astype(float)>=2.5)]
#

#mm,um,mo,uo=Match1to1Summary(um,reySum,6.0)
#print len(mm),len(um),len(mo),len(uo)
##um=um.astype(float)
#x=um[:,1].astype(float)
#y=um[:,2].astype(float)
#print sorted(um[np.where((y>3650)&(y<3750)&(x>500)&(x<600))][:,0])
#quit()
##plt.plot(reySum[:,1],reySum[:,2],'go')
##plt.plot(mm[:,1],mm[:,2],'bo')
##plt.plot(um[:,1],um[:,2],'ro')
##plt.hist(mm[:,6].astype(int),bins=50,color='g')
##plt.hist(um[:,6].astype(int),bins=50)
##plt.show()
#np.savetxt('UnmatchedSCSN.csv',um,delimiter=',',fmt='%s')
#quit()

## Make an array for the mag calculation, new IDs with the better locations (give old times to match pick file)
#arr=mm
#arr[:,0]=mo[:,0]
#arr[:,5]=mo[:,5]
#arr=arr[np.argsort(arr[:,5])]
#np.savetxt('SCSNEvent_2016Nov16IDs.csv',arr,delimiter=',',fmt='%s')
#quit()

#
#plt.hist(uo[:,6].astype(float),bins=40)
#plt.hist(mo[:,6].astype(float),bins=40)
#plt.show()

# Get the args which relate to each event type (noise/real/tele/...)
#noiseArgs,unsureArgs,realArgs=[],[],[]
#farArgs,dupArgs,scsnArgs=[],[],[]
#scsnIDs=[]
#realCount=0
#for i,aID in enumerate(reySum[:,0].astype(int)):
#    if aID in noiseIDs:
#        noiseArgs.append(i)
#    elif aID in unsureIDs:
#        unsureArgs.append(i) ## Unsure is currently added to noise
#    elif aID in dupIDs:
#        dupArgs.append(i)
#    elif aID in farIDs:
#        farArgs.append(i)
#    elif aID in mo[:,0].astype(int):
#        scsnArgs.append(i)
#        scsnIDs.append(aID)
#        realCount+=1
#    else:
#        realArgs.append(i)
#        realCount+=1
#print realCount
#allNoiseArgs=np.array(noiseArgs+unsureArgs+dupArgs+farArgs)
#allRealArgs=np.array(realArgs+scsnArgs)
#eveSumProb=reySum[:,6].astype(float)

# For ALL events, ie. sumEveProb>=2.5
#np.savetxt('2016Nov16_NoiseEvents.csv',reySum[allNoiseArgs],delimiter=',',fmt='%s')
#quit()

## Show histogram vs. SumEveProb
#plt.figure(figsize=(15,12))
#plt.hist([eveSumProb[scsnArgs],eveSumProb[realArgs],eveSumProb[allNoiseArgs]],
#         bins=np.arange(2.5,17.5,0.1), histtype='bar', stacked=True,
#         color=['b','g','r'],label=['scsn','real & new','noise'])
#plt.xlabel('Summed Event Probability')
#plt.ylabel('Event Count')
#plt.xlim(2.5,17.5)
#plt.legend()
##plt.yscale('log')
#plt.savefig('SumEveProbHist.png')
#plt.close()
  
#noiseArgs,unsureArgs,realArgs=np.array(noiseArgs),np.array(unsureArgs),np.array(realArgs)
#farArgs,dupArgs,scsnArgs=np.array(farArgs),np.array(dupArgs),np.array(scsnArgs)
#scsnIDs=np.array(scsnIDs)
#percReal=100.0*realCount/float(len(reySum))
#gain=realCount/171.0

# Plot histogram vs. timestamp
t1,t2=UTCDateTime('2014-02-02').timestamp,UTCDateTime('2014-02-19').timestamp
tbounds=np.arange(t1,t2+0.1,3600.00)
reyTimes=np.array([UTCDateTime(aEntry[5]).timestamp for aEntry in reySum])
shelTimes=np.array([UTCDateTime(aEntry[5]).timestamp for aEntry in shelSum])

# Plot the number of counts in each bin
count1,bin1=np.histogram(reyTimes,bins=tbounds)
count2,bin2=np.histogram(shelTimes,bins=tbounds)
plt.title('Time Bin Count Compare')
plt.plot([0,160],[0,160],'k')
plt.plot(count1,count2,'ko',ms=2)
plt.ylabel('ShelCount')
plt.xlabel('ReyCount')
plt.axes().set_aspect('equal')
plt.show()

plt.figure(figsize=(15,12))
#plt.hist([reyTimes],
#         bins=tbounds, histtype='bar', stacked=True,
#         color=['b'],label=['ReyRaw'])
plt.hist(reyTimes,bins=tbounds,label='ReyRaw Catalog',alpha=0.6)
plt.hist(shelTimes,bins=tbounds,label='Shel Catalog',alpha=0.6)
plt.xlim(t1,t2)
plt.xlabel('Timestamp (s)')
plt.ylabel('Event Count')
#plt.title(str(len(reyTimes))+' Total Events, '+'{:.1f}'.format(percReal)+'% Real, '+'{:.1f}'.format(gain)+'x MoreRealEvents')
plt.legend()
#plt.ylim(0,100)
plt.show()
#plt.savefig('EveTimeHist.png')
plt.close()

# Show which times have the high potential for overlap
#count=0
#for i in range(len(reyTimes)-1):
#    if reyTimes[i+1]-reyTimes[i]<15:
#        print reySum[i][0],reySum[i+1][0],reyTimes[i+1]-reyTimes[i]
#        count+=1
#print count


#
##plt.plot(staInfo[:,2],staInfo[:,1],'g^',markersize=8,label='NX Station')
###plt.plot(staInfo2[:,1],staInfo2[:,2],'c^',markersize=8,label='OGS Station')
#dists=[]
#for i in range(len(mm)):
#    plt.plot([mm[i][1],mo[i][1]],[mm[i][2],mo[i][2]],'k')
#    plt.plot(mo[i][1],mo[i][2],'bo')
#    plt.plot(mm[i][1],mm[i][2],'ro')
#    dist=np.sum((mm[i][1:4].astype(float)-mo[i][1:4].astype(float))**2)**0.5
#    dists.append(dist)
###    if dist>13:
###        print mm[i][0],mo[i][0],dist
#print np.median(dists)
#rect = mpatches.Rectangle([375, 3550], 325, 375, fc="none",ec='k',label='Grid stack boundary')
#plt.gca().add_patch(rect)
#plt.plot([],[],'bo',label='New Catalog')
#plt.plot([],[],'ro',label='SCSN Catalog')
##plt.ylim(3550, 3925)
##plt.xlim(375,700)
#plt.title('2016Jun09-2016Jun15 Events')
#plt.xlabel('Easting (km)')
#plt.ylabel('Northing (km)')
#plt.legend(numpoints=1)
#plt.axes().set_aspect('equal')
#plt.show()

#print shelSum.shape
#print reySum.shape
## See where most event are using density plot
#xGridInfo=[np.min(reySum[:,1].astype(float))-0.5,np.max(reySum[:,2].astype(float))+0.5,0.5]
#yGridInfo=[np.min(reySum[:,2].astype(float))-0.5,np.max(reySum[:,2].astype(float))+0.5,0.5]
##data=(shelSum[:,np.array([1,2,3])].T).astype(float)
#data=(reySum[:,np.array([1,2,3])].T).astype(float)
#xData,yData,zData=downSampleData(data,0,1,[2],xGridInfo,yGridInfo,[len])
#xGrid,yGrid,zGrid=fillGridData(xData,yData,zData[0][0],emptyAsNaN=True)
#minX,maxX=np.min(xGrid),np.max(xGrid)
#minY,maxY=np.min(yGrid),np.max(yGrid)
#plt.imshow(zGrid,interpolation='none',origin='lower',
#           extent=[minX-0.5*xGridInfo[2],maxX+0.5*xGridInfo[2],
#                   minY-0.5*yGridInfo[2],maxY+0.5*yGridInfo[2]])
#plt.plot(staInfo[:,2],staInfo[:,1],'g^',ms=9)
#plt.title('New Catalog Automatic')
##plt.gca().add_patch(rect)
##plt.plot(reySum[:,1],reySum[:,2],'ro',ms=2)
#plt.colorbar()
#plt.show()
#plt.close()
