#basic nn to classify number(digits 0 , 1 , 2 , 3 ....)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
X_axis = []
Y_axis = []
Accuracy = []
Total = [] 
class NUmber_classify(nn.Module):
    def __init__(self , num_classes=10):
        super(NUmber_classify , self).__init__()
        self.layer1 = nn.Linear(784 , 128 )
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(128 , num_classes)
       

    def forward(self , X):
        X = self.layer1(X)
        X = self.activation1(X)
        X = self.layer2(X)
        return X


transform = transforms.Compose([
    transforms.ToTensor(),                   
    transforms.Normalize((0.5,), (0.5,))      
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = NUmber_classify(num_classes= 10)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr = 0.001)
for i in range(10): #i can do more epochs but it will take a lot of time.But after 10 epochs i get an accuracy that is appropriate.
    model.train()
    correct = 0
    total = 0
    loss_init = 0
    for images , labels in train_loader:
        images = images.view(images.size(0) , -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs , labels)
        loss.backward()
        optimizer.step()
        loss_init += loss.item()
        _, predicted = torch.max(outputs , 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = 100 * (correct / total)
    print(accuracy)
    print(loss_init)
    X_axis.append(i)
    Y_axis.append(loss_init)
    Correct.append(correct)
    Total.append(total)
    Accuracy.append(accuracy)

plt.subplot(1 , 2 , 1)
plt.plot(X_axis , Y_axis)
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1 ,2 ,2)
plt.plot(X_axis , Accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()        
        
 


