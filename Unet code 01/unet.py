import torch
import torch.nn as nn

# Helper function to define a double convolution block
def double_conv(in_c, out_c):
    """
    Creates a sequential block of two Conv2D layers, each followed by ReLU activation.
    
    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        
    Returns:
        nn.Sequential: A sequential block containing two Conv2D layers and ReLU activations.
    """
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),  # First 3x3 convolution
        nn.ReLU(inplace=True),                 # Activation for the first convolution
        nn.Conv2d(out_c, out_c, kernel_size=3),  # Second 3x3 convolution
        nn.ReLU(inplace=True),                 # Activation for the second convolution
    )
    return conv

# Function to crop a tensor to match the size of the target tensor
def crop_img(tensor, target_tensor):
    """
    Crops a tensor to match the height and width of the target tensor by removing borders.
    
    Args:
        tensor (torch.Tensor): The tensor to crop.
        target_tensor (torch.Tensor): The tensor with the desired size.
        
    Returns:
        torch.Tensor: The cropped tensor.
    """
    target_size = target_tensor.size()[2]  # Target height and width
    tensor_size = tensor.size()[2]        # Original height and width
    delta = tensor_size - target_size     # Difference in size
    delta = delta // 2                    # Half of the difference
    return tensor[:, :, delta : tensor_size - delta, delta : tensor_size - delta]

# Define the UNet model class
class UNet(nn.Module):

    def __init__(self):
        """
        Initializes the UNet model with an encoder and decoder structure.
        """
        super(UNet, self).__init__()

        # Encoder: Downsampling layers
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2x2 pooling
        self.down_conv_1 = double_conv(1, 64)      # First block: Input to 64 channels
        self.down_conv_2 = double_conv(64, 128)    # Second block: 64 to 128 channels
        self.down_conv_3 = double_conv(128, 256)   # Third block: 128 to 256 channels
        self.down_conv_4 = double_conv(256, 512)   # Fourth block: 256 to 512 channels
        self.down_conv_5 = double_conv(512, 1024)  # Fifth block: 512 to 1024 channels

        # Decoder: Upsampling layers
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # Upsample from 1024 to 512 channels
        self.up_conv_1 = double_conv(1024, 512)  # Concatenate and process

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)

        # Final output layer
        self.out = nn.Conv2d(64, 2, kernel_size=1)  # Output 2 channels (e.g., for binary segmentation)

    def forward(self, image):
        """
        Defines the forward pass of the UNet model.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2, height, width).
        """
        # Encoder: Contracting path
        x1 = self.down_conv_1(image)  # First convolution block
        x2 = self.max_pool_2x2(x1)    # Downsample
        x3 = self.down_conv_2(x2)     # Second convolution block
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)     # Bottleneck

        # Decoder: Expanding path
        x = self.up_trans_1(x9)       # Upsample
        y = crop_img(x7, x)           # Crop to match size
        x = self.up_conv_1(torch.cat((x, y), 1))  # Concatenate and process
        
        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat((x, y), 1))
        
        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat((x, y), 1))
        
        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat((x, y), 1))

        x = self.out(x)  # Final output layer
        print(x.size())  # Debug print to check output size

        return x

# Main function to test the model
if __name__ == "__main__":
    # Generate a random input image with 1 channel and size 572x572
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    print(model(image))  # Forward pass and print the output
