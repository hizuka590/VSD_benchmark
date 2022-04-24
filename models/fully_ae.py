import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import dataset.dataloader as VSD_data

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True



##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 123
learning_rate = 0.005
num_epochs = 20
batch_size = 64

# Architecture
num_features = 173056
num_hidden_1 = 256


##########################
### VSD DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset,valid_dataset= VSD_data.get_dataset()

train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

valid_loader = DataLoader(
        valid_dataset, batch_size=64, shuffle=True
    )


# # Checking the dataset
# for images, labels in train_loader:
#         print('Image batch dimensions:', images.shape)
#         print('Image label dimensions:', labels.shape)
#         img = images[0]
#         lab = labels[0]
#
#
#         f = plt.figure()
#         f.add_subplot(2, 2, 1)
#         plt.imshow(img.permute(1, 2, 0), aspect='auto')
#         f.add_subplot(2, 2, 2)
#         plt.imshow(lab.permute(1, 2, 0), aspect='auto')
#         plt.show()
#         break


##########################
### MODEL
##########################

class Autoencoder(torch.nn.Module):

        def __init__(self, num_features):
                super(Autoencoder, self).__init__()

                ### ENCODER
                self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
                # The following to lones are not necessary,
                # but used here to demonstrate how to access the weights
                # and use a different weight initialization.
                # By default, PyTorch uses Xavier/Glorot initialization, which
                # should usually be preferred.
                self.linear_1.weight.detach().normal_(0.0, 0.1)
                self.linear_1.bias.detach().zero_()

                ### DECODER
                self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
                self.linear_1.weight.detach().normal_(0.0, 0.1)
                self.linear_1.bias.detach().zero_()

        def forward(self, x):
                ### ENCODER
                encoded = self.linear_1(x)
                encoded = F.leaky_relu(encoded)

                ### DECODER
                logits = self.linear_2(encoded)
                decoded = torch.sigmoid(logits)

                return decoded


torch.manual_seed(random_seed)
model = Autoencoder(num_features=num_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
for epoch in range(num_epochs):
        for batch_idx, (features, targets) in enumerate(train_loader):

                # don't need labels, only the images (features)
                features = features.view(-1, 416 * 416).to(device)

                ### FORWARD AND BACK PROP
                decoded = model(features)
                cost = F.binary_cross_entropy(decoded, features)
                optimizer.zero_grad()

                cost.backward()

                ### UPDATE MODEL PARAMETERS
                optimizer.step()

                ### LOGGING
                if not batch_idx % 50:
                        print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                              % (epoch + 1, num_epochs, batch_idx,
                                 len(train_loader), cost))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))