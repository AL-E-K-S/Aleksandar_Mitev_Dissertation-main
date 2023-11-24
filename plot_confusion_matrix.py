import itertools
import numpy as np
import matplotlib.pyplot as plt 
import os 
# function adapted from: deeplizard (2019). CNN Confusion Matrix with Pytorch – Neural Network Programming. YouTube  https://www.youtube.com/watch?v=0LhiS6yu2qQ  
# adaptation: removing the code to create a normalized confusion matrix and making it save the figure
def plot_confusion_matrix(cm,classes,title='Confusion Matrix',cmap=plt.cm.Blues):
    
    print(cm)
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j],fmt)),
        horizontalalignment="center",
        color="white" if cm[i,j] > thresh else "black"
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    

    

