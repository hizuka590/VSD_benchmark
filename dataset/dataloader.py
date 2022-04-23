from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import glob
import numpy
import random
import albumentations as A
import albumentations.pytorch
# import utils.joint_transforms as joint_transforms
# from torchvision import transforms

#######################################################
#               Define Transforms
#######################################################
train_transforms = A.Compose(
    [
        A.Resize(416,416),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Resize(416, 416),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)

#######################################################
#               Define train,val,test sets path
#######################################################
def get_alldata_path(train_image_path, type='train'):
    list_image_paths = []  # to store image paths in list
    # 1.
    # get all the paths from train_data_path and append image paths and class to to respective lists
    # eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
    # eg. class -> 26.Pont_du_Gard
    for data_path in glob.glob(train_image_path + '/*'):
        # print(data_path)
        list_image_paths.append(glob.glob(data_path + '/*'))
        # print('appended: ',glob.glob(data_path + '/*'))
        # break

    list_image_paths = [item for sublist in list_image_paths for item in sublist] # flatten needed for shuffle because ori is a 2d list
    # print('ori:',train_image_paths)
    random.shuffle(list_image_paths)
    # print('after:', train_image_paths)

    print('{}_image_path example number:{},random sample:{} '.format(type, len(list_image_paths),random.choice(list_image_paths)))
    return list_image_paths

#######################################################
#               Define Dataset Class
#######################################################

class VSD_DataSet(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(image_filepath)
        label = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)

        print('iamge shape: {} label shape: {}'.format(image.shape,label.shape))

        return image, label



if __name__ == "__main__":
    # app.run(main)
    train_image_path = '/opt/sdb/polyu/VSD_dataset/train/images'
    valid_image_path = '/opt/sdb/polyu/VSD_dataset/test/images'
    #######################################################
    #                  Create Dataset
    #######################################################

    train_dataset = VSD_DataSet(get_alldata_path(train_image_path),train_transforms)
    valid_dataset = VSD_DataSet(get_alldata_path(valid_image_path, type='val'),test_transforms)  # test transforms are applied
    # test_dataset = VSD_DataSet(test_image_paths)

    #######################################################
    #                  Define Dataloaders
    #######################################################

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=64, shuffle=True
    )

    # test_loader = DataLoader(
    #     test_dataset, batch_size=64, shuffle=False
    # )
    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        img = images.squeeze()
        plt.imshow(img, cmap="gray")
        plt.show()
        break

