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
class ELU_Hybrid_3D_2D(layers.Module): 
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

        super(ELU_Hybrid_3D_2D, self).__init__() # 
        
        self.cn1 = layers.Conv3d(in_channels = 30,out_channels = 8,kernel_size = 7, padding = 3) # the images are in  coluor channel and 29 tensors 
        self.cn2 = layers.Conv3d(in_channels = 8,out_channels = 16,kernel_size = 5, padding = 2)
        self.cn3 = layers.Conv3d(in_channels = 16,out_channels=32, kernel_size = 3, padding = 1)
        
        self.cn4_2d = layers.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        


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
        output = F.elu(output)

        
        output = self.cn2(output)
        output = F.elu(output)

        
        output = self.cn3(output)
        output = F.elu(output) # activaltion function

    
        
        output = output.reshape(output.size(0), 32,64,64)
        
        
        output = self.cn4_2d(output)
        
        output = F.elu(output)
        

        #output = output.view(-1) # 
        output = output.flatten(start_dim=1)

        output = self.fc5(output)
        output = F.elu(output)
        output = self.dropout5(output)


        output = self.fc6(output)
        output = F.elu(output)
        output = self.dropout6(output)
        

        output = self.fc7(output)

        #output = F.softmax(output,dim = 0) # since there are 2 classes 

        return output


def train(model,device,the_dataloader, optim,epochs):
    # Input model device the_dataloader optim epoch
    #
    #
    # 
    # Output model 
    
    
    
    model.train() # setd the model to training mode 
    
    for epoch in range(epochs):
        print("Current epoch" + str(epoch))

        start_time = time.time()

        loss_list = []    
        batch_list = []

        predicted_y = []
        actual_y = []

        for batch_index, (X,y) in enumerate(the_dataloader):
            #X,y = X.to(device), y.to(device) # X is the input y is teh ground truth
            
            prediction = model(X) # gets the predictions made by teh model 
            
            loss = layers.CrossEntropyLoss()
            loss_result =  loss(prediction,y) # to be chenged to a different one since it does softmax 
            #loss =  # calculates the loss

            loss_result.backward() # updates the weights 
            optim.step()
            optim.zero_grad() # sets the gradients to 0 

            loss_list.append(loss_result)
            batch_list.append(batch_index)
            # code adapted from christianbernecker meduum.com ...
            # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-with-tensorboard-and-pytorch-3344ad5e7209
            prediction  = (torch.max(torch.exp(prediction), 1)[1]).data.cpu().numpy()
            predicted_y.extend(prediction) 
            
            y = y.data.cpu().numpy()
            actual_y.extend(y) # adds the actual result to the tensor with ground truths
        
        


            if (batch_index*30) % 30 == 0: # 
                print(batch_index)
                training_result_format = 'batch:({:.0f})|loss:({:.4f}) '.format(batch_index,loss_result)
                print(training_result_format)

        end_time = time.time()
        execution_time = end_time - start_time
        ELU_3D_2D_Hybrid_Time = open(('ELU_3D_2D_Hybrid_Training_Time.txt'), "w" ) 
        ELU_3D_2D_Hybrid_Time.write("Training time:" + str(execution_time) + "s" + "\n") 
        ELU_3D_2D_Hybrid_Time.close() 

        ELU_3D_2D_Hybrid_Testing_Loss = open(("ELU_3D_2D_Hybrid_Training_Loss.csv"), "w" ,newline='') # opens the csv
        csvWritingFileObject = csv.writer(ELU_3D_2D_Hybrid_Testing_Loss) # creates an instance of teh class that will write to teh csv
        rowOfData = [batch_list,loss_list] # image details to be added to teh csv
        csvWritingFileObject.writerow(rowOfData) 
        
        # predicted and actual values for confusion matrix
        ELU_3D_2D_Hybrid_Conf_Met_Values = open(('ELU_3D_2D_Hybrid_Conf_Met_Values_Training.txt'), "w" ) 
        ELU_3D_2D_Hybrid_Conf_Met_Values.write("Predicted values: " + str(predicted_y) + "\n" + "Actual values:" + str(actual_y)) 
        ELU_3D_2D_Hybrid_Conf_Met_Values.close() 

    return model

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


        loss_list.append(loss_result)
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
    ELU_3D_2D_Hybrid_Time = open(('ELU_3D_2D_Hybrid_Time.txt'), "w" ) 
    ELU_3D_2D_Hybrid_Time.write("Testing time:" + str(execution_time) + "s" + "\n") 
    ELU_3D_2D_Hybrid_Time.close() 

    ELU_3D_2D_Hybrid_Testing_Loss = open(("ELU_3D_2D_Hybrid_Testing_Loss.csv"), "w" ,newline='') # opens the csv
    csvWritingFileObject = csv.writer(ELU_3D_2D_Hybrid_Testing_Loss) # creates an instance of teh class that will write to teh csv
    rowOfData = [batch_list,loss_list] # image details to be added to teh csv
    csvWritingFileObject.writerow(rowOfData) 
    
    # predicted and actual values for confusion matrix
    ELU_3D_2D_Hybrid_Conf_Met_Values = open(('ELU_3D_2D_Hybrid_Conf_Met_Values_Testing.txt'), "w" ) 
    ELU_3D_2D_Hybrid_Conf_Met_Values.write("Predicted values: " + str(predicted_y) + "\n" + "Actual values:" + str(actual_y)) 
    ELU_3D_2D_Hybrid_Conf_Met_Values.close() 
    



ELUModel = ELU_Hybrid_3D_2D()




optim = optimizers.SGD(ELUModel.parameters(),lr=0.0005)# transfer learning paper > used 0.0005 for transfer learning 




training_data, testing_data = torch.utils.data.random_split(dataset, [42,18])

torch.manual_seed(0) # by setting a seed all random numbers generated can be made the same for all models

load_training_data = DataLoader(dataset=training_data, batch_size=12, shuffle=3) 
load_testing_data = DataLoader(dataset=testing_data, batch_size=12, shuffle=3)



ELU_3D_2D_Hybrid_ELU = train(ELUModel, 'cpu',load_training_data,optim,epochs=1) # ?

test(ELU_3D_2D_Hybrid_ELU, 'cpu',load_testing_data)

torch.save(ELU_3D_2D_Hybrid_ELU,"3D_2D_HybridWeights_ELU.pt") # creates a file for saving the trained model 





