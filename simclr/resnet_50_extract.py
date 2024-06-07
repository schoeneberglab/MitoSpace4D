import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights


### Input : 3 channel batch image (3, 356, 256)
### Ouput : (256, 64, 64)
class ResNet50Extract(nn.Module):
    def __init__(self, layer_name="layer1", bottleneck=2, activation="relu", in_channels=256): #In channels should match with output of hooked layer
        super(ResNet50Extract, self).__init__()
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self._freeze_params()
        self.extracted_output = None

        self._register_hook(layer_name, bottleneck, activation)
        self.downsample = nn.Conv2d(in_channels, 1, kernel_size=3,stride=1, padding=1)

    def _freeze_params(self):
        for param in self.resnet50.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        _ = self.resnet50(x) # Creates extracted output = (-1, 256, 64, 64)
        conv_out = self.downsample(self.extracted_output) # (-1, 1, 64, 64)
        return conv_out.squeeze(1) # (-1, 64, 64)
    
    def _register_hook(self, layer_name, bottleneck, activation):
        def hook_fn(module, input, output):
            self.extracted_output = output

        hooked_layer = self._get_hooked_layer(layer_name, bottleneck, activation)
        hooked_layer.register_forward_hook(hook_fn)

    def _get_hooked_layer(self, layer_name, bottleneck, activation):
        layer = getattr(self.resnet50, layer_name)

        assert isinstance(layer, nn.Sequential) # Should be bottleneck layer
        layer = layer[bottleneck]

        if not activation:
            return layer
        
        activation_layer = getattr(layer, activation)
        return activation_layer

# Example usage
if __name__ == '__main__':
    extractor = ResNet50Extract()
    dummy_input = torch.randn(1, 3, 256, 256)
    intermediate_output = extractor(dummy_input)
    print(f'Intermediate output shape: {intermediate_output.shape}')