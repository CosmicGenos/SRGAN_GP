import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
'''
This is the class that process and tranforms the images. you know we have get the 96x96 patches with High
Resalution image and 4x downsampled versions of that exact patch from Low resalution image 

1)is inherit from dataset class, so we can put in Dataloader, __getitem__ have
2)there is 3 pre process in this
    *)>create_image_path_pairs
    -----this takes the High resalution and low resalution image paths, we can give mutiple paths since we use mutiple datasets
    it zip them up take both frist elements from the path arrays(HR paths,LR paths) because those are contains smaller versions and 
    larger versions of pictures. thos lisdir going to get all the names of the given path. then using list compherhnsion we 
    put picture path and picture name in a list like this[(high_path,high_name),(Low_path,low_name)] . so we can get full path for a image.
    so we can put it on the load image which is to load that image. then transformation method takes that image then transform and convert
    to tensor. Lr_image,Hr_image wise
'''

class SRdataset(Dataset):
    def __init__(self,High_resalution_paths,Low_resalution_paths,Scale = 4,crop_size = 96,is_traning =True):
        super(SRdataset,self).__init__()

        self.High_paths = High_resalution_paths
        self.Low_paths = Low_resalution_paths
        self.scale = Scale
        self.crop_size = crop_size
        self.is_traning = is_traning
        self.path_and_image_pairs = self.create_image_path_pairs()
        self.High_Resalution_image_transformations = A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()])
        self.Low_Resalution_image_transformations = A.Compose([A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), ToTensorV2()])


    def create_image_path_pairs(self):

        pairs = []

        for high_path,low_path in zip(self.High_paths,self.Low_paths):

            hr_file_names = sorted(os.listdir(high_path))
            lr_file_names = sorted(os.listdir(low_path))

            if len(hr_file_names) != len(lr_file_names):
                raise RuntimeError(f"Mismatched file counts in {high_path} ({len(hr_file_names)}) and {low_path} ({len(lr_file_names)})")
            
            dir_pairs_with_imges = list(zip(
                                        [(high_path,hr_file_name) for hr_file_name in hr_file_names],
                                        [(low_path,lr_file_name) for lr_file_name in lr_file_names])
                                        )
            
            pairs.extend(dir_pairs_with_imges)

        return pairs
    
    def load_image(self,path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not Load the image of path : {path}")
        return  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def Transformation(self,high_resalution_image,Low_resalution_image):
         
        if not self.is_traning:
            high_resalution_Tensor = self.High_Resalution_image_transformations(image = high_resalution_image)["image"]
            Low_resalution_Tensor = self.Low_Resalution_image_transformations(image = Low_resalution_image)["image"]

            return Low_resalution_Tensor,high_resalution_Tensor
        
        if self.is_traning:

            h,w = high_resalution_image.shape[:2]
            x = np.random.randint(0,h - self.crop_size +1)
            y = np.random.randint(0,w - self.crop_size + 1)

            low_res_crop_size = self.crop_size // self.scale
            low_res_crop_x = x // self.scale
            low_res_crop_y = y // self.scale

            highres_crop = high_resalution_image[y:y+self.crop_size, x:x+self.crop_size]
            lowres_crop = Low_resalution_image[low_res_crop_y:low_res_crop_y+low_res_crop_size, low_res_crop_x:low_res_crop_x+low_res_crop_size]

            do_flip = np.random.random() < 0.5
            if do_flip:
                highres_crop = np.fliplr(highres_crop).copy() 
                lowres_crop = np.fliplr(lowres_crop).copy()

            high_resalution_Tensor = self.High_Resalution_image_transformations(image=highres_crop)['image']  
            Low_resalution_Tensor = self.Low_Resalution_image_transformations(image=lowres_crop)['image']  
            
            return Low_resalution_Tensor, high_resalution_Tensor 
        
    
    def __getitem__(self, idx):

        (hr_path,hr_name),(lr_path,lr_name) = self.path_and_image_pairs[idx]
        
        hr_path = os.path.join(hr_path, hr_name)
        lr_path = os.path.join(lr_path, lr_name)
        
        hr_image = self.load_image(hr_path)
        lr_image = self.load_image(lr_path)

        return self.Transformation(hr_image, lr_image) #return lower_res , Hig_res

    def __len__(self):
        return len(self.path_and_image_pairs)

             

            

# # Example usage
# dataset = SRDataset(
#     high_resolution_paths=['/path/to/hr1', '/path/to/hr2'],
#     low_resolution_paths=['/path/to/lr1', '/path/to/lr2'],
#     scale=4,
#     crop_size=96,
#     is_training=True
# )

# # Create DataLoader
# dataloader = DataLoader(
#     dataset,
#     batch_size=16,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )