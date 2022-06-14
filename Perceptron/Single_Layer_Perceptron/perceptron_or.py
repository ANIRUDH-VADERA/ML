import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,classification_report


x = [ [-1,-1], [-1,1], [1,-1], [1,1]]
x_array = np.asarray(x)
# expected outputs (AND port is the product of each entry
out = np.asarray([-1,1,1,1])

m = 2
n = 4

w=[]
for i in range(m):
    w.append(0.5)
theta = 0
bias = -1
learning_rate = 1

print("Initial Weights : " , w)

sum = 0
epochs = 10
e = 0
error = [0,0,0,0]
E = 100
pred = []

while E!=0 and e < epochs:
    pred = []
    E = 0
    for i in range(n):
        sum = 0
        for j in range(m):
            sum += x[i][j]*w[j]
        sum = sum + bias
        if(sum > theta):
            output = 1
        else:
            output = -1
        pred.append(output)
        error[i] = out[i] - output
        for j in range(m):
            if(out[i] != output):
                w[j] = w[j] + learning_rate*(out[i]-output)*x[i][j]
                bias = bias + learning_rate*out[i]
    for i in error:
        E = E + abs(i)
    e = e + 1
    print("After " , e , "Epochs : ")
    print("Prediction : " , pred)
    print("Actual Values : " , out)
    print("Error : " , error)


print("The Final Weights are : ", w)
print("The Final Bias is : " , bias)
print("The Final Error is : " , "Error Array : ",error,"Error : ",E)

# Checking the accuracy of our model
print('Accuracy: ',accuracy_score(out,pred))
print('Precision: %.3f' % precision_score(out, pred,average='micro'))
print('Recall: %.3f' % recall_score(out, pred,average='micro'))

# Our Model Report
print('*************** Evaluation on Our Model ***************')
print('Accuracy Score: ', accuracy_score(out,pred))
# Look at classification report to evaluate the model
print(classification_report(out, pred))
print('--------------------------------------------------------')
print("")