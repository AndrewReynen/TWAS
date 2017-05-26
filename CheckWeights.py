import numpy as np
import matplotlib.pyplot as plt

# Unpack the weights, and see where the higher values cluster
TryNo='02'
W0=np.load('stack_W0.npy')
B0=np.load('stack_B0.npy')
print B0
classes=['Stacked','Unstacked','Noise']

yLabels=['DOP1','DOP2','DOP3',
         'F1-V0','F1-V1','F1-V2','F1-V3','F1-V4','F1-V5','F1-V6','F1-V7','F1-V8',
         'F1-H0','F1-H1','F1-H2','F1-H3','F1-H4','F1-H5','F1-H6','F1-H7','F1-H8',
         'F2-V0','F2-V1','F2-V2','F2-V3','F2-V4','F2-V5','F2-V6','F2-V7','F2-V8',
         'F2-H0','F2-H1','F2-H2','F2-H3','F2-H4','F2-H5','F2-H6','F2-H7','F2-H8']
maxColor=0.15
minColor=-0.15
numTimeSamples=8
numFeatures=39


# Plot
i=1
fig=plt.figure(figsize=(10.5,6))
for j in range(len(W0[0])):
    title=classes[j]
    W=W0[:,j].reshape((2*numFeatures,numTimeSamples))
    W_S=W[numFeatures:,:]
    W_P=W[:numFeatures,:]
    # Plot P-weights
    plt.subplot(1,7,i)
    plt.title(title+'\nP-weights')
    plt.imshow(W_P,interpolation='none',cmap = plt.get_cmap('RdYlBu_r'), vmin=minColor, vmax=maxColor)
    if i==1:
        plt.gca().set_yticks(range(39))
        plt.gca().set_yticklabels(yLabels,fontsize=10)
    else:
        plt.gca().set_yticks([])
    
    i+=1
    # Plot S-weights
    plt.xlabel('Time Index',fontsize=14)
    plt.yticks()
    plt.subplot(1,7,i)
    plt.title(title+'\nS-weights')
    plt.imshow(W_S,interpolation='none',cmap = plt.get_cmap('RdYlBu_r'), vmin=minColor, vmax=maxColor)
    plt.gca().set_yticks([])
    plt.xlabel('Time Index',fontsize=14)
    if i==6:
        cax=fig.add_axes([0.81,0.115,0.04,0.77])
        cax.imshow(np.array([np.arange(minColor,maxColor,0.0025)]).T,origin='lower',
                   cmap = plt.get_cmap('RdYlBu_r'), vmin=minColor, vmax=maxColor,
                   aspect='auto',extent=[0,1,minColor,maxColor])
        cax.set_xticks([])
        cax.yaxis.tick_right()
    i+=1
plt.savefig('weight_'+TryNo+'.png')
plt.close()


