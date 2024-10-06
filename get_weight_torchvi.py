import torch 
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), "torchvision_vgg16_state.pt")