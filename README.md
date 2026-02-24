# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name:  LOKESH S
### Register Number: 21224240079
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)

        return x

```

```python
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
def train_model(model, train_loader, num_epochs=3):
  print('Name: LOKESH S ')
  print('Register Number:    212224240079 ')
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # write your code here

      print('Name: LOKESH S ')
      print('Register Number:    212224240079 ')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch
<img width="374" height="216" alt="image" src="https://github.com/user-attachments/assets/4a30e14a-70d6-46bb-bc4f-359aae521532" />


### Confusion Matrix

<img width="814" height="711" alt="image" src="https://github.com/user-attachments/assets/9e0cd179-7001-48b4-b2cf-68999ddcd5ff" />


### Classification Report

<img width="815" height="540" alt="image" src="https://github.com/user-attachments/assets/1d71ddf6-268f-4440-8559-ae8ffc1a5911" />




### New Sample Data Prediction

<img width="450" height="495" alt="image" src="https://github.com/user-attachments/assets/3d20c1c1-7d7b-4be4-8e1e-5b4ea7589f6f" />


## RESULT
Include your result here.
