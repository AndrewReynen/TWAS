#from sklearn import svm
from Classification_TensorFlow_withBurst import loadData
import numpy as np

xTrain,xCv,xTest,yTrain,yCv,yTest,nClass=loadData()
yTrain=np.argmax(yTrain,axis=1)
yTest=np.argmax(yTest,axis=1)

#print 'Train'
#svc=svm.SVC(kernel='rbf')
##svc=svm.LinearSVC()
#svc.fit(xTrain,yTrain)
#
## Test saving the model
from sklearn.externals import joblib
#joblib.dump(svc, 'filename.pkl')
svc2=joblib.load('filename.pkl') 

print 'Predict'
pred=svc2.predict(xTest)

outArr=[]
for i in range(nClass):
    row=[]
    for j in range(nClass):
        count=len(np.where((yTest==i)&(pred==j))[0])
        row.append(count)
    outArr.append(row)
print np.array(outArr)
