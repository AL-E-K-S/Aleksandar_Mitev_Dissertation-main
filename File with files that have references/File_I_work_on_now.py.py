import torch.nn.functional as F
import torch.nn as layers
import torch.optim as optimizers
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from videoDataLoaderTest import dataset
from torch.utils.data import DataLoader, Dataset
import seaborn as sn
import time
import csv
import os
from plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt 


from Regular_3D_2D_Hybrid_Extra_Conv2d import Regular_Plus_2dModel
from ELU_3D_2D_Hybrid_Extra_Conv2d import ELU_Plus_2dModel
from ELU_3D_2D_Hybrid import ELUModel

class Hybrid_3D_2D(layers.Module): # layers.Module is what the Hybrid_3D_2d will inherit methods from
    # Input layers.Module 
    #
    #
    #
    # Output
    def __init__(self):
        # Input self 
        #
        #
        # 
        # Output 

        super(Hybrid_3D_2D, self).__init__() # 
        
        self.cn1 = layers.Conv3d(in_channels = 30,out_channels = 8,kernel_size = 7, padding = 3) # the images are in  coluor channel and 29 tensors 
        self.cn2 = layers.Conv3d(in_channels = 8,out_channels = 16,kernel_size = 5, padding = 2)
        self.cn3 = layers.Conv3d(in_channels = 16,out_channels=32, kernel_size = 3, padding = 1)
        #
        #self.cn1_additional = layers.Conv3d(in_channels = 32,out_channels=64, kernel_size = 3, padding = 1)
        #self.cn2_additional = layers.Conv3d(in_channels = 64,out_channels=128, kernel_size = 3, padding = 1)
        #self.cn3_additional = layers.Conv3d(in_channels = 128,out_channels=256, kernel_size = 3, padding = 1)
        # self.cn4_2d = layers.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3, padding = 1) # padding for new layers need to be calculated 

        self.cn4_2d = layers.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1) # padding for new layers need to be calculated 
        


        # size > reduce the input from 3d > 2d layer
        # 2d layer output to be flattened 
        
        self.dropout5 = layers.Dropout(0.4)
        self.fc5 = layers.Linear(64 * 64 * 64 ,256)
        #dropout > 0.4
        self.dropout6 = layers.Dropout(0.4)
        self.fc6 = layers.Linear(256,128)
        self.fc7 = layers.Linear(128,2)
        #dropout > 0.4

    def forward(self,x):

        output = self.cn1(x)
        output = F.relu(output)

        
        output = self.cn2(output)
        output = F.relu(output)

        
        output = self.cn3(output)
        output = F.relu(output) # activaltion function

    
        
        output = output.reshape(output.size(0), 32,64,64)
        
        
        output = self.cn4_2d(output)
        
        output = F.relu(output)
        

        #output = output.view(-1) # 
        output = output.flatten(start_dim=1)

        output = self.fc5(output)
        output = F.relu(output)
        output = self.dropout5(output)


        output = self.fc6(output)
        output = F.relu(output)
        output = self.dropout6(output)
        

        output = self.fc7(output)

        #output = F.softmax(output,dim = 0) # since there are 2 classes 

        return output





# plot the graph


y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    #ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    #ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./lossGraphs', 'train.jpg'))



def train(model,device,the_dataloader, optim,epochs,dataset_size,phase,batch_size):
    # Input model device the_dataloader optim epoch to 
    #
    #
    # 
    # Output model 
    
    training_data_predictions = torch.tensor([]) # creates a tensor for storing the predictions of teh model 
    training_data_actual_values = torch.tensor([])
    
    
    model.train() # setd the model to training mode 
    
    for epoch in range(epochs):



        running_loss = 0.0 # sets loss for this epoch to 0
        running_corrects = 0.0 # sets actual truths to 0 for this epoch

        print("Current epoch" + str(epoch))

        #start_time = time.time()

        loss_list = []    
        batch_list = []

        predicted_y = []
        actual_y = []

        for batch_index, (X,y) in enumerate(the_dataloader):
            #X,y = X.to(device), y.to(device) # X is the input y is teh ground truth
            
            optim.zero_grad() # sets the gradients to 0 
            prediction = model(X) # gets the predictions made by teh model
            print(y)
            training_data_predictions = torch.cat((training_data_predictions, prediction),dim=0) # combines the predictions of the model for this batch and the ones from before
            training_data_actual_values = torch.cat((training_data_actual_values, y),dim=0) # adds the tensors with the actual values

            print(X.shape)
            # parts of code below adapted from 
            if X.size == 4:
                current_batch_size, channels, height, width = X.shape
            else:
                current_batch_size, framesFed ,channels, height, width = X.shape


            if current_batch_size < batch_size: # skips smaller batch to plot the graph
                continue


            _, preds = torch.max(prediction.data, 1)
            loss = layers.CrossEntropyLoss()

            
            loss_result = loss(prediction,y) # to be chenged to a different one since it does softmax 
            
            
            del X

            
            #loss =  # calculates the loss

            loss_result.backward() # updates the weights 
            optim.step()
            
            running_loss+=loss_result.item() * current_batch_size

            del loss
            running_corrects+=float(torch.sum(preds == y.data))
             
            
            #batch_list.append(batch_index)
            # code adapted from christianbernecker meduum.com ...
            # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-with-tensorboard-and-pytorch-3344ad5e7209
            #prediction  = (torch.max(torch.exp(prediction), 1)[1]).data.cpu().numpy()
            #predicted_y.extend(prediction) 
            
            #y = y.data.cpu().numpy()
            #actual_y.extend(y) # adds the actual result to the tensor with ground truths
        
        


            if (batch_index*30) % 30 == 0: # 
                print(batch_index)
                training_result_format = 'batch:({:.0f})|loss:({:.4f}) '.format(batch_index,loss_result)
                print(training_result_format)




    epoch_loss = running_loss/dataset_size # ;loss fpr this epoch
    epoch_acc = running_corrects / dataset_size # 
    y_loss[phase].append(epoch_loss)
    y_err[phase].append(1.0  - epoch_acc)

    draw_curve(epoch)

    Regular_3D_2D_Hybrid_Testing_Loss = open(("Regular_3D_2D_Hybrid_Training_Loss.csv"), "w" ,newline='') # opens the csv
    csvWritingFileObject = csv.writer(Regular_3D_2D_Hybrid_Testing_Loss) # creates an instance of teh class that will write to teh csv
    rowOfData = [batch_list,loss_list] # image details to be added to teh csv
    csvWritingFileObject.writerow(rowOfData) 

    return model,training_data_predictions,training_data_actual_values



def test(model,device,the_dataloader):
   
    start_time = time.time()

    loss_list = []    
    batch_list = []

    predicted_y = []
    actual_y = []
    

    for batch_index, (X,y) in enumerate(the_dataloader):
        X,y = X.to(device), y.to(device) # X is the input y is teh ground truth
        model.eval() # sets model to evaluation mode 
        torch.no_grad()
        prediction = model(X) # gets the predictions made by teh model 
        
        loss = layers.CrossEntropyLoss() 
        loss_result = loss(prediction,y)  # calculates the loss 

        batch_list.append(batch_index)
        # code adapted from christianbernecker meduum.com ...
        # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-with-tensorboard-and-pytorch-3344ad5e7209
        
        prediction  = (torch.max(torch.exp(prediction), 1)[1]).data.cpu().numpy()
        predicted_y.extend(prediction) 
            
        y = y.data.cpu().numpy()
        actual_y.extend(y) # adds the actual result to the tensor with ground truths
        
        
        if batch_index % 10 == 0:
            print(batch_index)
            testing_result_format = 'batch:({:.0f})|loss:({:.4f}) '.format(batch_index,loss_result)
            print(testing_result_format)

            # add the batch number and training loss

    

    end_time = time.time()
    execution_time = end_time - start_time
    Regular_3D_2D_Hybrid_Time = open(('Regular_3D_2D_Hybrid_Time.txt'), "w" ) 
    Regular_3D_2D_Hybrid_Time.write("Testing time:" + str(execution_time) + "s" + "\n") 
    Regular_3D_2D_Hybrid_Time.close() 


ReLUModel = Hybrid_3D_2D()




optim = optimizers.SGD(ReLUModel.parameters(),lr=0.0005)# transfer learning paper > used 0.0005 for transfer learning 




training_data, testing_data = torch.utils.data.random_split(dataset, [42,18])

torch.manual_seed(0) # by setting a seed all random numbers generated can be made the same for all models

load_training_data = DataLoader(dataset=training_data, batch_size=12, shuffle=3) 
load_testing_data = DataLoader(dataset=testing_data, batch_size=12, shuffle=3)



def make_train_results_confusion_matrix(training_predictions,training_data_actual_values):
    # code below adapted from:
    # adaptation > turning it into a function and changing number of classes of this project
    predicted_class = training_predictions.argmax(dim=1) # gets highest predicted probability for a class
    stacked_predictions = torch.stack((training_data_actual_values,predicted_class),dim=1)
    stacked_predictions[0].tolist()

    confusion_matrix_tensor = torch.zeros(2,2, dtype=torch.int32)

    for pred in stacked_predictions:
        j, k = pred.tolist()
        confusion_matrix_tensor[j,k] = confusion_matrix_tensor[j,k] + 1
        



training_results = train(ReLUModel, 'cpu',load_training_data,optim,1,len(training_data),'train',batch_size=12) # ?



pretrained3D_2D_Hybrid = training_results[0]


training_predictions = (training_results[1]).detach().argmax(dim=1)
training_data_actual_values = (training_results[2])

classes = ["No Violence","Violence"]

print("Training predictions")
print(training_predictions)
print("Actual values")
print(training_data_actual_values)

confusion_mat = confusion_matrix(training_predictions,training_data_actual_values)

## Model Confusion Matrix Plotting Section
plot_confusion_matrix(confusion_mat,classes,title='Confusion Matrix1')

print(type(confusion_mat))

plt.figure(figsize=(10,10))
plt.savefig(os.path.join('./Confusion Matrices', 'reg_model_confuse_matrix1.png'))

plot_confusion_matrix(confusion_mat,classes,title='Confusion Matrix2')

print(type(confusion_mat))

plt.figure(figsize=(10,10))
plt.savefig(os.path.join('./Confusion Matrices', 'reg_model_confuse_matrix2.jpg'))

## end of Plotting Section

test(pretrained3D_2D_Hybrid, 'cpu',load_testing_data)





torch.save(pretrained3D_2D_Hybrid,"pretrained3D_2D_HybridWeights.pt") # creates a file for saving the trained model 




# train and test with ELU

pretrained3D_2D_Hybrid_ELU = train(ELUModel, 'cpu',load_training_data,optim,epochs=1) # ?

test(pretrained3D_2D_Hybrid_ELU, 'cpu',load_testing_data)

torch.save(pretrained3D_2D_Hybrid_ELU,"pretrained3D_2D_HybridWeights_ELU.pt") # creates a file for saving the trained model 

# train and test with 




pretrained3D_2D_Hybrid_Regular_Plus_2d = train(Regular_Plus_2dModel, 'cpu',load_training_data,optim,epochs=1) # ?

test(pretrained3D_2D_Hybrid_Regular_Plus_2d, 'cpu',load_testing_data)

torch.save(pretrained3D_2D_Hybrid_Regular_Plus_2d,"pretrained3D_2D_HybridWeights_Regular_Plus_2d.pt") # creates a file for saving the trained model 




pretrained3D_2D_Hybrid_ELU_Plus_2d = train(ELU_Plus_2dModel, 'cpu',load_training_data,optim,epochs=1) # ?


pretrained3D_2D_Hybrid_ELU_Plus_2d = test(pretrained3D_2D_Hybrid_ELU_Plus_2d, 'cpu',load_testing_data)

torch.save(pretrained3D_2D_Hybrid_ELU_Plus_2d,"pretrained3D_2D_HybridWeights_ELU_Plus_2d.pt") # creates a file for saving the trained model 

