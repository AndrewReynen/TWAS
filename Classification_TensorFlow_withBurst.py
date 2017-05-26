import tensorflow as tf
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
import matplotlib.pyplot as plt
import numpy as np


# Create Y data
def createYdata(xGroups):
    numClasses=len(xGroups)
    yData=np.zeros((0,numClasses),dtype=int)
    for i,arr in enumerate(xGroups):
        yVals=np.zeros((len(arr),numClasses))
        yVals[:,i]=1
        yData=np.vstack((yData,yVals))
    return yData,numClasses

# Split data between train,cv, and test...
# ...for all the groups
def createSetIdxs(yData,pTrain,pCv,pTest):
    idxs=[[],[],[]] # Train, Cv, Test
    for i in range(len(yData[0])):
        classIdxs=np.where(yData[:,i]==1)[0]
        # Shuffle randomly, and append to the different sets
        np.random.shuffle(classIdxs)
        m=len(classIdxs)
        idxs[0]+=list(classIdxs[:int(m*pTrain)])
        idxs[1]+=list(classIdxs[int(m*pTrain):int(m*(pTrain+pCv))])
        idxs[2]+=list(classIdxs[int(m*(pTrain+pCv)):])
    for i in range(len(idxs)):
        idxs[i]=np.array(idxs[i])
    return idxs


# Function to initialize weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Function to initialize bias values
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def loadData():
    # Load X data
    xStack=np.load('./stackX.npy')[:30000,:]
    xUnstack=np.load('./unstackX.npy')[:30000,:]
    xNoise=np.load('./noiseX.npy')[:30000,:]
    # Create Y data
    xGroups=[xStack,xUnstack,xNoise]
    xAll=np.vstack((arr for arr in xGroups))
    yAll,nClass=createYdata(xGroups)
    # Randomly suffle data into train, cv and test
    iTrain,iCv,iTest=createSetIdxs(yAll,0.6,0.2,0.2)


    xTrain,yTrain=xAll[iTrain],yAll[iTrain]
    xCv,yCv=xAll[iCv],yAll[iCv]
    xTest,yTest=xAll[iTest],yAll[iTest]
    # Calculate the mean/stdev based off the training set, and apply to the test set
    xStdev=np.std(xTrain,axis=0)
    xMean=np.mean(xTrain,axis=0)  
    
    xTrain=(xTrain-xMean)/xStdev
    xCv=(xCv-xMean)/xStdev
    xTest=(xTest-xMean)/xStdev
    return xTrain,xCv,xTest,yTrain,yCv,yTest,nClass


def tfLogReg(xTrain,xCv,xTest,yTrain,yCv,yTest,nClass):
    numFeatures=len(xTrain[0])
#    numHiddenNodes=50
    sess = tf.InteractiveSession()
    # Input layer
    A0=tf.placeholder(tf.float32, [None, numFeatures])
    W0=weight_variable([numFeatures, nClass])
    B0=bias_variable([nClass])
#    W0=weight_variable([numFeatures, numHiddenNodes])
#    B0=bias_variable([numHiddenNodes])
    # Hidden layer
#    A1=tf.nn.sigmoid(tf.matmul(A0, W0)+B0)
#    W1=weight_variable([numHiddenNodes, numClasses])
#    B1=bias_variable([numClasses])
    # Output layer
#    A2=tf.nn.sigmoid(tf.matmul(A0, W0)+B0) 
#    A2=tf.nn.sigmoid(tf.matmul(A1, W1)+B1) 
    A2=tf.nn.softmax(tf.matmul(A0, W0) + B0) # Trying out softmax & cross-entropy
#    A2=tf.nn.softmax(tf.matmul(A1, W1) + B1) # Trying out softmax & cross-entropy
    Y=tf.placeholder(tf.float32, [None, nClass]) # The known output
    # Cost
#    cost=(tf.reduce_mean(-tf.reduce_sum(Y*tf.log(A2)+(1.0-Y)*tf.log(1-A2),reduction_indices=[1])))#+
#          0.01*(tf.reduce_sum(tf.abs(W0))))
#          0.01*(tf.reduce_sum(tf.abs(W0))+tf.reduce_sum(tf.abs(W1))))
#          0.01*(tf.reduce_sum(W0**2)))
#          0.01*(tf.reduce_sum(W0**2)+tf.reduce_sum(W1**2)))
    cost=(tf.reduce_mean(-tf.reduce_sum(Y * tf.log(A2), reduction_indices=[1]))+
          0.00*(tf.reduce_sum(W0**2))) # Trying out softmax & cross-entropy
    train_step = tf.train.GradientDescentOptimizer(0.10).minimize(cost)
#    tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    #%% Start loading data into the model and iterating for best solution
    numSteps=250
    costTrains,costTests=[],[]
    accValTrains,accValTests=[],[]
    for i in range(numSteps):
        print i
        if i==100:
            train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
        train_step.run(feed_dict={A0: xTrain, Y: yTrain})
        # Check the accuracy & cost
        correct_prediction = tf.equal(tf.argmax(A2,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accValTests.append(accuracy.eval(feed_dict={A0: xTest, Y: yTest}))
        accValTrains.append(accuracy.eval(feed_dict={A0: xTrain, Y: yTrain}))
        costTests.append(cost.eval(feed_dict={A0: xTest, Y: yTest}))
        costTrains.append(cost.eval(feed_dict={A0: xTrain, Y: yTrain}))
    # Plot accuracy and cost
    iterNums=range(1,numSteps+1)
    plt.plot(iterNums,accValTests,'r^',label='TestAcc')
    plt.plot(iterNums,accValTrains,'b^',label='TrainAcc')
    plt.plot(iterNums,costTests/np.max(costTests),'ro')
    plt.plot(iterNums,costTrains/np.max(costTests),'bo')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()
    # Save the weights and bias values
#    finW0=W0.eval(sess)
#    finB0=B0.eval(sess)
#    finB1=B1.eval(sess)
#    finW1=W1.eval(sess)
#    for aLabel,aEntry in [['W0',finW0],#['W1',finW1],
#                          ['B0',finB0],#['B1',finB1],
#                          ['xMean',xMean],['xStdev',xStdev]]:
#        np.save('stack_'+aLabel,aEntry)
    
    prob=A2.eval(feed_dict={A0: xTest})
    # Look at the confusion matrix
    pred=np.argmax(prob,axis=1)
    actual=np.argmax(yTest,axis=1)
    outArr=[]
    for i in range(nClass):
        row=[]
        for j in range(nClass):
            count=len(np.where((actual==i)&(pred==j))[0])
            row.append(count)
        outArr.append(row)
    print np.array(outArr)
    
    
if __name__=='__main__':
    tfLogReg(*loadData())