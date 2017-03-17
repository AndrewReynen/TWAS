import numpy as np
import matplotlib.pyplot as plt

# Unpack the weights, and see where the higher values cluster
TryNo='03'
W0=np.load('./UsedParams/W0.npy')
B0=np.load('./UsedParams/B0.npy')
Wni=W0[:,0]
Weq=W0[:,1]
Wrv=W0[:,2]
yLabels=['DOP1','DOP2','DOP3',
         'F1-V0','F1-V1','F1-V2','F1-V3','F1-V4','F1-V5',
         'F1-H0','F1-H1','F1-H2','F1-H3','F1-H4','F1-H5',
         'F2-V0','F2-V1','F2-V2','F2-V3','F2-V4','F2-V5',
         'F2-H0','F2-H1','F2-H2','F2-H3','F2-H4','F2-H5']
maxColor=0.05
minColor=-0.05
numTimeSamples=6
numFeatures=27

print 'B values: [Noise, Earthquake, Reversed]'
print B0

# For Noise
Wni=Wni.reshape((2*numFeatures,numTimeSamples))
Wni_P=Wni[:numFeatures,:]
Wni_S=Wni[numFeatures:,:]
# For earthquakes
Weq=Weq.reshape((2*numFeatures,numTimeSamples))
Weq_P=Weq[:numFeatures,:]
Weq_S=Weq[numFeatures:,:]
# For reversed earthquakes
Wrv=Wrv.reshape((2*numFeatures,numTimeSamples))
Wrv_P=Wrv[:numFeatures,:]
Wrv_S=Wrv[numFeatures:,:]

# Plot
i=1
fig=plt.figure(figsize=(10.5,6))
for W_S,W_P,Label,title in [[Weq_S,Weq_P,'EQ','Earthquake'],
                            [Wrv_S,Wrv_P,'RV','Reversed'],
                            [Wni_S,Wni_P,'NI','Noise']]:
    plt.subplot(1,7,i)
    plt.title(title+'\nP-weights')
    plt.imshow(W_P,interpolation='none',cmap = plt.get_cmap('bwr'), vmin=minColor, vmax=maxColor)
    if i==1:
        plt.gca().set_yticks(range(numFeatures))
        plt.gca().set_yticklabels(yLabels,fontsize=10)
    else:
        plt.gca().set_yticks([])
    
    i+=1
#    plt.ylabel('Feature Name')
    plt.xlabel('Time Index',fontsize=14)
#    plt.colorbar()
    plt.yticks()
    plt.subplot(1,7,i)
    plt.title(title+'\nS-weights')
    plt.imshow(W_S,interpolation='none',cmap = plt.get_cmap('bwr'), vmin=minColor, vmax=maxColor)
    plt.gca().set_yticks([])
    plt.xlabel('Time Index',fontsize=14)
    if i==6:
        cax=fig.add_axes([0.81,0.115,0.04,0.77])
        cax.imshow(np.array([np.arange(minColor,maxColor,0.0025)]).T,origin='lower',
                   cmap = plt.get_cmap('bwr'), vmin=minColor, vmax=maxColor, aspect='auto', extent=[0,1,minColor,maxColor])
        cax.set_xticks([])
        cax.yaxis.tick_right()
#        cax.yaxis.set_label_position("right")
#        cax.set_ylabel('Weight')
#        cax.set_yticks([])
    i+=1
plt.savefig('Weights_'+TryNo+'.png')
plt.close()
#plt.show()


