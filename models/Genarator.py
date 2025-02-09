'''
in SRGAN generator has three main parts:
1. Feature extraction	
    -> uses covelutional layers to extract features from the input image
2. feature analysis
    -> uses residual blocks to analyze the features extracted in the previous step
3. Image Upsampling
    -> uses upsampling layers to increase the size of the image
'''

import torch
import torch.nn as nn
import time

class convelutional_block(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 64,kernel_size = 9 ,padding = 4, stride = 1):
        super(convelutional_block, self).__init__()
        self.convo1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.PR = nn.PReLU()

    def forward(self, x):
        x = self.convo1(x)
        x = self.PR(x)
        return x
    
class Resedual_connection(nn.Module):
    def __init__(self,kernel_size = 3,channels = 64,padding = 1, stride = 1):
        super(Resedual_connection, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size,stride, padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        return x
    
class Post_residual_convolution(nn.Module):
    def __init__(self, in_channels = 64,  kernel_size = 3, padding = 1, stride = 1):
        super(Post_residual_convolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x

class Upsample_block(nn.Module):
    def __init__(self, in_channels = 64, out_channels = 256, kernel_size = 3, padding = 1, stride = 1,upscale_factor = 2):
        super(Upsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor )
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
    
class Genarator(nn.Module):
    #im not going to add input parameters here because it would be really long, if any thing should change from default values, you have to chance it on the class
    def __init__(self,num_res_blocks = 16, upsampling_factor = 4):
        super(Genarator, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.conv1 = convelutional_block()
        reasuidual_layer_list =[Resedual_connection() for _ in range(num_res_blocks)]
        self.residual_layers = nn.Sequential(*reasuidual_layer_list)
        self.post_residual_convolution = Post_residual_convolution()
        self.upsampling = nn.Sequential(
            Upsample_block(upscale_factor=2),
            Upsample_block(upscale_factor=2)
        )
        self.final_layer = nn.Conv2d(64, 3, 9, 1, 4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.residual_layers(x)
        x = self.post_residual_convolution(x)
        x += residual
        x = self.upsampling(x)
        x = self.final_layer(x)
        x = self.tanh(x)
        return x

#testing the model

if __name__ == '__main__':
    x = torch.randn(1,3,256,256)
    t1 = time.time()
    model = Genarator()
    t2 = time.time()
    print(model(x).shape)
    print(f"Time taken to create model to infernce {t2-t1}")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    

'''	

outputs :torch.Size([1, 3, 256, 256])
Time taken to create model to infernce 0.015674114227294922


time is changed by every time i runs, and im using a cpu to run this code, so it will be slow, if you have a gpu it will be faster, but should push model in to gpu before running the model
and if you are using cpu, please dont input larger size it will stuck your pc, it happend to me when i try to input 1024x1024 image, 
'''

'''	
the biggest problem i have with SRGAN is it suppose to be independent of input size, but i had douts, most of them with descriminator , but in genarator its all convelutional
and, it not using any fully connected layers, and pixel shuffle layer is used to increase the size of the image, so it should be independent of input size
and all the time we use kernal we use padding to keep the size of the image same, so it should be independent of input size


'''	