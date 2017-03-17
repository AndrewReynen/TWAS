import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys,os
sys.path.insert(0, '/home/andrewr/Desktop/Python')
from Reynen_Functions import getPCA


eqDir='./FeatureArchive_NCEDC_npy'
noiseDir1='./FeatureArchive_Noise_npy'
Vp,Vs=4.7,2.7
Dp,Ds=0.2,0.2
preArriv=1 # Number of samples before arrival to use for classification
postArriv=3 # Number of sampling after arrival to use for classification
nBand=8 # Number of bands used when extracting F1 and F2 features

# Load features
def loadData():
    # Read in the data, note that features are row-wise
    x=[] # Input features 
    y=[] # 1 for eq, 0 for blast
    stas=[] # to note which parent event/station it came from
    ids=[] # the event ID
    hyps=[] # hypocentral distance
    missCount=0
    for aFolder,aClass,addToID in [[noiseDir1,[1,0,0],0],
                                   [eqDir,[0,1,0],0],
                                   [eqDir,[0,0,1],100000000]]:
        print 'Loading ',aFolder
        for aFile in (os.listdir(aFolder)):
            data=np.load(aFolder+'/'+aFile)
            hypDist=data[0][0]
            tp=(hypDist-Vp*Dp)/Vp
            argP=int(round(np.argmin(np.abs(data[1]-tp))))-preArriv
            ts=(hypDist-Vs*Ds)/Vs
            argS=int(round(np.argmin(np.abs(data[1]-ts))))-preArriv
            # For the reversed class
            if aFolder==eqDir and aClass[2]==1:
                xToAdd=np.concatenate((data[2:,argS:argS+(preArriv+postArriv)].flatten(),
                                       data[2:,argP:argP+(preArriv+postArriv)].flatten()))
            else:
                xToAdd=np.concatenate((data[2:,argP:argP+(preArriv+postArriv)].flatten(),
                                       data[2:,argS:argS+(preArriv+postArriv)].flatten()))
            if len(xToAdd)!=(preArriv+postArriv)*2*(3+4*nBand):
                missCount+=1
                continue
            if np.sum(np.isfinite(xToAdd))!=len(xToAdd):
                missCount+=1
                continue
            x.append(xToAdd)
            y.append(aClass)
            stas.append(aFile.split('_')[1])
            ids.append(int(aFile.split('_')[0])+addToID)
            hyps.append(hypDist)
    x=np.array(x)
    y=np.array(y)
    ids=np.array(ids)
    stas=np.array(stas)
    hyps=np.array(hyps)
    print str(missCount)+' entries were skipped as too short, or contained a non-finite number'
    return x,y,ids,stas,hyps

# Function to initialize weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Function to initialize bias values
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main():
    #%% Load and preprocess the data
    xAll,yAll,ids,stas,hyps=loadData()
    print '---'
    print xAll.shape
    
    # Initial List of ids
    noiseIDs=np.arange(100000,103000)
    eqIDs=np.genfromtxt('NCEDC_Trimmed_UTM11.csv',delimiter=',',dtype=str)[:,0].astype(int) 
    # Shuffle them randomly
    randIdxEq=range(len(eqIDs))
    randIdxNoise=range(len(noiseIDs))
    np.random.shuffle(randIdxEq)
    np.random.shuffle(randIdxNoise)
    eqIDs=eqIDs[randIdxEq]
    noiseIDs=noiseIDs[randIdxNoise]
    lEq=len(eqIDs)
    lNi=len(noiseIDs)
    # 60%,20%,20% Train, CV, Test sets
    trainEqIDs=eqIDs[:int(lEq*0.6)]
    cvEqIDs=eqIDs[int(lEq*0.6):int(lEq*0.8)]
    testEqIDs=eqIDs[int(lEq*0.8):]
    trainNoiseIDs=noiseIDs[:int(lNi*0.6)]
    cvNoiseIDs=noiseIDs[int(lNi*0.6):int(lNi*0.8)]
    testNoiseIDs=noiseIDs[int(lNi*0.8):] 
    # Split into training, cross-validation and test sets
    trainRows,cvRows,testRows=[],[],[]
    for i,aID in enumerate(ids):
        if (aID%100000000 in trainEqIDs) or (aID%100000000 in trainNoiseIDs):
            trainRows.append(i)  
        elif (aID%100000000 in cvEqIDs) or (aID%100000000 in cvNoiseIDs):
            cvRows.append(i)
        elif (aID%100000000 in testEqIDs) or (aID%100000000 in testNoiseIDs):
            testRows.append(i)
        else:
            print aID,'was not added to train nor test'
            
    trainRows=np.array(trainRows)
    cvRows=np.array(cvRows)
    testRows=np.array(testRows)
    xTrain,yTrain,staTrain=xAll[trainRows],yAll[trainRows],stas[trainRows]
    xCV,yCV,staCV=xAll[cvRows],yAll[cvRows],stas[cvRows]
    xTest,yTest,iTest,staTest=xAll[testRows],yAll[testRows],ids[testRows],stas[testRows]
    # Calculate the mean/stdev based off the training set, and apply to the test set
    xStdev=np.std(xTrain,axis=0)
    xMean=np.mean(xTrain,axis=0)  
    xTrain=(xTrain-xMean)/xStdev
    xCV=(xCV-xMean)/xStdev
    xTest=(xTest-xMean)/xStdev
    print xTrain.shape,xCV.shape,xTest.shape
    pred=np.argmax(yAll,axis=1)
#    print len(np.where((pred==0))[0])
#    print len(np.where((pred==1))[0])
#    print len(np.where((pred==2))[0])
#    quit()

    
    #%% Set up tensor flow model
    def trainWeights(regVal,numSteps,initStepSize):
        numFeatures=len(xTrain[0])
    #    numHiddenNodes=50
        numClasses=yTrain.shape[1]
    #    print numClasses,numFeatures
        sess = tf.InteractiveSession()
        # Input layer
        A0=tf.placeholder(tf.float32, [None, numFeatures])
        W0=weight_variable([numFeatures, numClasses])
        B0=bias_variable([numClasses])
        A2=tf.nn.softmax(tf.matmul(A0, W0) + B0) # Trying out softmax & cross-entropy
        Y=tf.placeholder(tf.float32, [None, numClasses]) # The known output
        # Cost
        cost=(tf.reduce_mean(-tf.reduce_sum(Y * tf.log(A2), reduction_indices=[1]))+
              regVal*(tf.reduce_sum(tf.abs(W0)))) # Trying out softmax & cross-entropy
    #          0.01*(tf.reduce_sum(W0**2))
        train_step = tf.train.GradientDescentOptimizer(initStepSize).minimize(cost)
        tf.initialize_all_variables().run()
        #%% Start loading data into the model and iterating for best solution
        costTrains,costCVs=[],[]
        accValTrains,accValCVs=[],[]
        for i in range(numSteps):
#             # Decrease the step size slightly with each iteration... 
#             # ... which decreases to about 30% of original value at final step
#            stepSize=initStepSize*np.exp(-0.25*(5.0/numSteps*i))
#            train_step = tf.train.GradientDescentOptimizer(stepSize).minimize(cost)
            train_step.run(feed_dict={A0: xTrain, Y: yTrain})
            # Check the accuracy & cost
            correct_prediction = tf.equal(tf.argmax(A2,1), tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accValCVs.append(accuracy.eval(feed_dict={A0: xCV, Y: yCV}))
            accValTrains.append(accuracy.eval(feed_dict={A0: xTrain, Y: yTrain}))
            costCVs.append(cost.eval(feed_dict={A0: xCV, Y: yCV}))
            costTrains.append(cost.eval(feed_dict={A0: xTrain, Y: yTrain}))
        # Plot accuracy and cost
        iterNums=range(1,numSteps+1)
        plt.plot(iterNums,accValCVs,'r^',label='CVAcc')
        plt.plot(iterNums,accValTrains,'b^',label='TrainAcc')
        plt.plot(iterNums,costCVs/np.max(costCVs),'ro')
        plt.plot(iterNums,costTrains/np.max(costCVs),'bo')
        plt.legend(loc='upper left')
        plt.show()
        plt.close()
        # Save the weights and bias values
        finW0=W0.eval(sess)
        finB0=B0.eval(sess)
    #    finB1=B1.eval(sess)
    #    finW1=W1.eval(sess)
        for aLabel,aEntry in [['W0',finW0],#['W1',finW1],
                              ['B0',finB0],#['B1',finB1],
                              ['xMean',xMean],['xStdev',xStdev]]:
            np.save(aLabel,aEntry)
        
        prob=A2.eval(feed_dict={A0: xCV})
        pred=np.argmax(prob,axis=1)
        
        #%% See how many misclassifcations there are between noise / real events
        # Number of noise events classified as real
        actual=np.argmax(yCV,axis=1)
        noise2real=np.where(((actual==0)&(pred!=0)))[0]
        real2noise=np.where(((actual!=0)&(pred==0)))[0]
        staAcc=1-len(np.where((actual!=pred))[0])/float(len(yCV))
#        print 'Station Accuracy:',staAcc
        n2r,r2n=len(noise2real),len(real2noise)
        print staAcc,n2r,r2n,'Reg: '+str(regVal)+', numSteps: '+str(numSteps)+', initStepSize: '+str(initStepSize)
        sess.close()
    # Try a set of model parameters
    for regVal in [0.004]:
        for numSteps in [200]:
            for initStepSize in [0.2]:
                trainWeights(regVal,numSteps,initStepSize)
    quit()
#    weights = np.ones_like(hypTest)/len(hypTest)
#    hypTest=hyps[testRows]
#    # Calculate the counts within each bin
#    histAll,bins=np.histogram(hypTest,bins=np.arange(0,275.1,25))
#    histn2r,bins=np.histogram(hypTest[noise2real],bins=np.arange(0,275.1,25))
#    histr2n,bins=np.histogram(hypTest[real2noise],bins=np.arange(0,275.1,25))
#    binCenters=(bins[1:]+bins[:-1])/2.0
#    # Plot the portion (relative to all inputs) of events within each bin
#    plt.hist(hypTest,bins=np.arange(0,275.1,25),alpha=0.7,color='grey',weights=weights)
#    plt.plot(binCenters,histn2r.astype(float)/histAll,'r',label='noise2real')
#    plt.plot(binCenters,histr2n.astype(float)/histAll,'g',label='real2noise')
#    plt.xlabel('Hypocentral distance (km)')
#    plt.ylabel('Portion (% of given bin - for lines)')
#    plt.legend()
#    plt.show()
    
#    # Print out the noise samples which were categorized as real at short distances
#    print histn2r[0],histAll[0]
#    for arg in noise2real:
#        if hypTest[arg]<25:
#            print iTest[arg],staTest[arg],prob[arg]
#    quit()
    
#    # Show the weights (maybe figure out a more meaninful way of plotting?)
#    unqEvents=np.unique(iTest)
#    Ws=W0.eval(sess)
#    plt.imshow(Ws,origin='lower') 
#    plt.colorbar()
#    plt.show()
#    plt.close()
    #%% Look at the classification probabilities
    print '---'
    count=0
    countEve=0
    unqEvents=np.unique(iTest)
    for anID in unqEvents:
        wantIdxs=np.where((iTest==anID))[0]
        actual=np.argmax(yTest[wantIdxs[0]])
        pred=np.argmax(np.sum(prob[wantIdxs],axis=0))
        countEve+=1
        if actual==pred:
            count+=1
        else:
            print 'wrong class: ',anID,np.sum(prob[wantIdxs],axis=0)
#            for anIdx in wantIdxs:
#                print staTest[anIdx],prob[anIdx]
#            print prob[wantIdxs]
#            print staTest[wantIdxs]
    print 'Network Accuracy: ',count/float(countEve),count,countEve
#        eveProbs=prob[wantIdxs]
#        evePred=pred[wantIdxs]
    
main()


## Attempt at making the noise not a factor in cost function for the real portion
#costNoise=tf.reduce_mean(-1*(Y[:,0]*tf.log(A2[:,0])+(1.0-Y[:,0])*tf.log(1-A2[:,0])))
#costReal=tf.reduce_mean(-tf.reduce_sum(Y[:,1:]*tf.log(A2[:,1:])+(1.0-Y[:,1:])*tf.log(1-A2[:,1:]),reduction_indices=[1]))
#sumRealProba=tf.reduce_sum(A2[:,1:],reduction_indices=[1])
#sumRealProba=tf.transpose(tf.concat(0,([sumRealProba],[sumRealProba])))
#normRealProba=tf.truediv(A2[:,1:],sumRealProba)
#costReal=tf.reduce_mean(-tf.reduce_sum(Y[:,1:]*tf.log(normRealProba)+(1.0-Y[:,1:])*tf.log(1-normRealProba),reduction_indices=[1]))
#cost=costNoise+costReal

#            # Add in specific ratios
#            dataTrim=np.hstack((data[2:,argP:argP+8],data[2:,argS:argS+8]))
#            Hf0,Hf1,Hf5,Hf6=dataTrim[3+0],dataTrim[3+1],dataTrim[3+5],dataTrim[3+6]
#            Vf0,Vf1,Vf5,Vf6=dataTrim[12+0],dataTrim[12+1],dataTrim[12+5],dataTrim[12+6]
#            V50,V60=Vf5/Vf0,Vf6/Vf0
#            V51,V61=Vf5/Vf1,Vf6/Vf1
#            H50,H60=Hf5/Hf0,Hf6/Hf0
#            H51,H61=Hf5/Hf1,Hf6/Hf1
#            ratios=np.concatenate((V50,V60,V51,V61,H50,H60,H51,H61))
            # Normalize these ratios between zero and one
#            ratios=np.exp(-1*ratios)
#            xToAdd=np.concatenate((xToAdd,ratios))

#    # Check if the new features are actually being given a high weight
#    for i in range(len(finW0[0])):
#        w=finW0[:,i]
#        arr=w[-64:].reshape(8,8)
#        plt.imshow(arr,interpolation='none')
#        plt.colorbar()
#        plt.show()