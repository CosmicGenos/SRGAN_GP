import torch
import torch.autograd as autograd

def gradient_penalty(critc,real,fake,device):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).to(device) #this created random values that, between 0 and 1, and we are going to use it as epsilon for all image pixels for all batch images
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad = True
    mixed_scores = critc(interpolated_images)
    gradient = autograd.grad(
        inputs=interpolated_images,#[batch_size,channels,width,height]
        outputs=mixed_scores, #[batch_size,1]
        grad_outputs=torch.ones_like(mixed_scores),# we do this because,we want to get gradients of all the batch images
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1) #we get the gradient of all the batch images,because this return the images pixels gradients respect to the mixed scores
    gradient_norm = gradient.norm(2, dim=1) # this is not weight norm, this is the norm of the gradient of the image pixels
    gradient_penalty = torch.mean((gradient_norm - 0.01) ** 2)
    return gradient_penalty