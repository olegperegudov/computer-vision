
import torchvision.models as models
import torch.nn as nn


def resnet18(output_layer=2):
    # load the pretrainded model
    model = models.resnet18(pretrained=True)
    # freeze all params
    for param in model.parameters():
        param.requires_grad = False
    # add new layer. It will NOT be freezed by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_layer)
    return model
