from obspy import read,UTCDateTime

#st=read('/home/andrewr/Downloads/fdsnws-dataselect_2016-12-14T22-07-16.mseed')
#st.plot()

startTime=UTCDateTime('2014-02-02 00:00:00').timestamp
endTime=UTCDateTime('2014-02-19 00:00:00').timestamp
import numpy as np
#outArr=[]
#for i,aTime in enumerate(np.arange(startTime,endTime,3600)):
#    i=str(i).zfill(6)
#    outArr.append([i,str(UTCDateTime(aTime))])
#np.savetxt('MM_ArchiveTimes.csv',np.array(outArr,dtype=str),fmt='%s',delimiter=',')

# Make the noise times
info1=np.genfromtxt('./OldRelocShelly_UTM11.csv',delimiter=',',dtype=str)
info2=np.genfromtxt('./NewShelly_UTM11.csv',delimiter=',',dtype=str)
arr=np.vstack((info1,info2))
knownTimes=np.array([UTCDateTime(row[5]).timestamp for row in arr])

noiseCount=0
outArr=[]
times=[]
while noiseCount<3000:
    randTime=np.random.rand(1)[0]*(endTime-startTime)+startTime
    if np.min(np.abs(knownTimes-randTime))<30:
        continue
    print noiseCount
    times.append(randTime)
    outArr.append([str(noiseCount+100000),'321.4','4166.8','4.800','-999',str(UTCDateTime(randTime))])
    noiseCount+=1
np.savetxt('MM_NoiseSummary_UTM11.csv',np.array(outArr,dtype=str),fmt='%s',delimiter=',')
#import matplotlib.pyplot as plt
#plt.hist(times)
#plt.show()