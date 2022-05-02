import time
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from torch.nn import BCEWithLogitsLoss

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('logs/fashion_mnist_experiment_1')
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
num_epochs = 1
batch_size = 32

# Architecture
num_features = 173056
num_hidden_1 = 256
num_classes = 2


##########################
### VSD DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset,valid_dataset= VSD_data.get_dataset()

train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True
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
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads,alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


##########################
### MODEL
##########################

class Autoencoder(torch.nn.Module):

        def __init__(self, num_features):
                super(Autoencoder, self).__init__()

                ### ENCODER
                self.linear_r = torch.nn.Linear(num_features, num_hidden_1)
                self.linear_g = torch.nn.Linear(num_features, num_hidden_1)
                self.linear_b = torch.nn.Linear(num_features, num_hidden_1)
                # self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)



                ### DECODER
                self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)


        def forward(self, xr,xg,xb):
                ### ENCODER
                r = self.linear_r(xr)
                g = self.linear_g(xg)
                b = self.linear_b(xb)
                encoded = torch.add(r, g)
                encoded = torch.add(encoded, b)
                # encoded = self.linear_1(x)
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
        running_loss = 0.0
        for batch_idx, (features, targets) in enumerate(train_loader):

                # print(features.shape,targets.shape)

                # don't need labels, only the images (features)
                # print(features.shape)
                features_r = features[:,0,:,:].view(-1, 416 * 416).to(device)
                features_g = features[:, 1, :, :].view(-1, 416 * 416).to(device)
                features_b = features[:, 2, :, :].view(-1, 416 * 416).to(device)
                # print(features_r.shape)
                targets = targets.view(-1, 416 * 416).to(device)

                ### FORWARD AND BACK PROP
                decoded = model(features_r,features_g,features_b)

                cost = F.binary_cross_entropy(decoded, targets)
                optimizer.zero_grad()

                cost.backward()
                plot_grad_flow(model.named_parameters())


                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                running_loss = + cost.item() * features.shape[0]

                ### LOGGING
                if not batch_idx % 50:
                        print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                              % (epoch + 1, num_epochs, batch_idx,
                                 len(train_loader), cost))
        epoch_loss = running_loss / 4788
        writer.add_scalar('training loss',
                          epoch_loss)
                          # epoch * len(trainloader) + i)



        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

n_images = 15
image_width = 28

# fig, axes = plt.subplots(nrows=2, ncols=n_images,
#                          sharex=True, sharey=True, figsize=(20, 2.5))
# orig_images = features[:n_images]
# decoded_images = decoded[:n_images]
#
# for i in range(n_images):
#     for ax, img in zip(axes, [orig_images, decoded_images]):
#         curr_img = img[i].detach().to(torch.device('cpu'))
#         ax[i].imshow(curr_img.view((416, 416)), cmap='binary')