
import torchvision.models as models
import torch.nn as nn


def resnet18(output_layer):
    # load the pretrainded model
    model = models.resnet18(pretrained=True)
    # # freeze all params
    for param in model.parameters():
        param.requires_grad = False
    # add new layer. It will NOT be freezed by default
    num_ftrs = model.fc.in_features
    # add more fc layers for 5y output
    model.fc = nn.Linear(num_ftrs, output_layer)
    return model


def vgg(input_layer):
    model = models.vgg16(pretrained=True)
    # freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # change a few layers in classifier
    model.classifier[3] = nn.Linear(4096, 512, bias=True)
    model.classifier[6] = nn.Linear(512, input_layer, bias=True)

    return model


def alexnet(input_layer):
    model = models.alexnet(pretrained=True)
    # # freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # change a few layers in classifier
    model.classifier[3] = nn.Linear(4096, 2048, bias=True)
    model.classifier[4] = nn.Linear(2048, 512, bias=True)
    model.classifier[6] = nn.Linear(512, 2, bias=True)

    return model

# might use later
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def resnext50_32x4d(output_layer):
    # load the pretrainded model
    model = models.resnext50_32x4d(pretrained=True)
    # # freeze all params
    for param in model.parameters():
        param.requires_grad = False
    # add new layer. It will NOT be freezed by default
    num_ftrs = model.fc.in_features
    # add more fc layers for 5y output
    # model.fc = nn.Linear(num_ftrs, output_layer)
    # model.avgpool = Identity()
    model.fc = nn.Sequential(
                            nn.Linear(in_features=2048, out_features=1024, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_features=1024, out_features=512, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_features=512, out_features=output_layer, bias=True))
    return model


def wide_resnet50_2(output_layer):
    # load the pretrainded model
    model = models.resnext50_32x4d(pretrained=True)
    # # freeze all params
    for param in model.parameters():
        param.requires_grad = False
    # add new layer. It will NOT be freezed by default
    num_ftrs = model.fc.in_features
    # add more fc layers for 5y output
    model.fc = nn.Linear(num_ftrs, output_layer)
    return model

    
# model = vgg(5)
# model = alexnet(5)
# model = resnet18(5)
# model = resnet18_5(5)
# model = resnext50_32x4d_5(5)
# model = wide_resnet50_2(5)
# print(model)
