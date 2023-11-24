import torch.nn.functional as F
import torch.nn as layers
import torch.optim as optimizers
import torch

from LAD2000SUBSETDTloader import dataset,theLAD2000SubsetDataloader


print("file is being ran")

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
        




        
        self.cn1 = layers.Conv3d(in_channels = 1,out_channels = 8,kernel_size = 7, padding = 3) # the images are in  coluor channel and 29 tensors 
        self.cn2 = layers.Conv3d(in_channels = 8,out_channels = 16,kernel_size = 5, padding = 2)
        self.cn3 = layers.Conv3d(in_channels = 16,out_channels=32, kernel_size = 3, padding = 1)
        
        self.cn4_2d = layers.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        


        # size > reduce the input from 3d > 2d layer

        #RuntimeError: shape '[-1, 261]' is invalid for input of size 92588160  output = output.view(-1,29*3*3) 
        #RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2420, 288] < after feeding in filter 16 * 3 * 3 
        #RuntimeError: Given groups=1, weight of size [64, 32, 7, 7], expected input[32, 5, 66, 66] to have 32 channels, but got 5 channels instead > if the output from view function is unindented
        
        
        # 2d layer output to be flattened 
        
        self.dropout5 = layers.Dropout(0.4)
        self.fc5 = layers.Linear(1 * 64 * 64 ,256)
        #dropout > 0.4
        self.dropout6 = layers.Dropout(0.4)
        self.fc6 = layers.Linear(256,128)
        #dropout > 0.4

        #softmax layer > same number output as the number of classes 
        

    def forward(self,x):
        print("forward function reached")
        # Input self
        #
        #
        # 
        # Output 
        #print(x)
        print(x.shape)
        output = self.cn1(x)
        print(output.shape)
        output = self.cn2(output)
        print(output.shape)
        output = self.cn3(output)
        print(output.shape)
        ## debugging
        #output = output.view(-1,32*3*3)
        #output = output.view(-1,32*3*3)
        # reduced

        #output = output.view(-1,32 * 3 * 3) # reduction since it is going from 3d to 2d layer #  since 29 consecutive images are passed that will be the batch size

        #output = output.view(-1,5 * 3 * 3) # reduction since it is going from 3d to 2d layer #  since 29 consecutive images are passed that will be the batch size

        
        output = output.reshape(30, 32,64,64) # 30 is the batch size, 32 is the input from the previous layer and the 64s are teh pixel sizes
        output = self.cn4_2d(output)
        #output = output.reshape(output.size(0), 32,64,64)
        output = torch.flatten(output,1)
        print(output.shape)
        
        #output = torch.flatten(output,1) # reduction since it is going from 2d layer to a linear one

        output = self.fc5(output)
        output = self.fc6(output)
        output = F.softmax(6) # since there are 6 classes 

        return output




print(theLAD2000SubsetDataloader.__len__()) # 869 videos 









def train(model,device,the_dataloader, optim,epoch):
    # Input model device the_dataloader optim epoch
    #
    #
    # 
    # Output model 
    
    print("train function reached")
    model.train() # setd the model to training mode 
    



    for batch_index, (X,y) in enumerate(the_dataloader):
        #X,y = X.to(device), y.to(device) # X is the input y is teh ground truth
        
        prediction = model(X) # gets the predictions made by teh model 
        loss = layers.CrossEntropyLoss(prediction,y) # calculates the loss
        loss.backward() # updates the weights 
        optim.step()
        optim.zero_grad() # sets the gradients to 0 


        if batch_index % 10 == 0:
            print(batch_index)
            training_result_format = 'batch:({:.0f})|loss:({:.4f}) '.format(batch_index,loss)
            print(training_result_format)

            # add the batch number and training loss


        
    return model




 # ?



noPCAModel = Hybrid_3D_2D()
optim = optimizers.SGD(noPCAModel.parameters(),lr=0.0005)# transfer learning paper > used 0.0005 for transfer learning 
pretrained3D_2D_Hybrid = train(noPCAModel, 'cpu',theLAD2000SubsetDataloader,optim,epoch=2) # ?

torch.save(pretrained3D_2D_Hybrid,"pretrained3D_2D_HybridWeights.pt") # creates a file for saving the trained model 

print(dataset)

"""   
image_classes = ['Destroy','Falling','Fighting','Fire','Hurt','Thiefing']

model = Hybrid_3D_2D(image_classes = 6).to('cpu') # training with cpu ???
optim = optimizers.SGD(model.parameters(),lr=0.0005)# transfer learning paper > used 0.0005 for transfer learning
        #output = 


epoch_number = 2
for epoch in range(epoch_number):
    model.train() # sets model to trining mode 
    
    
    for index, (video,labels) in enumerate(dataset,theLAD2000SubsetDataloader):

        optimizer.zero_grad()

 self.cn1 = layers.Conv3d(29,8,3,padding=2) # the images are in rgb > 3 inchannels 8 since it is filter 3 allkernals > 3*3 
        self.cn2 = layers.Conv3d(8,16,3,padding=2)
        self.cn3 = layers.Conv3d(16,32,3)

        # size > reduce the input from 3d > 2d layer

        #RuntimeError: shape '[-1, 261]' is invalid for input of size 92588160  output = output.view(-1,29*3*3) 
        #RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [2420, 288] < after feeding in filter 16 * 3 * 3 
        #RuntimeError: Given groups=1, weight of size [64, 32, 7, 7], expected input[32, 5, 66, 66] to have 32 channels, but got 5 channels instead > if the output from view function is unindented
        self.cn4_2d = layers.Conv2d(5,64,3)        

"""

