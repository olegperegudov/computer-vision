
import torchvision.models as models
import torch.nn as nn


def resnet18(output_layer=2):
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


def vgg(input_layer=2):
    model = models.vgg16(pretrained=True)
    # freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # change a few layers in classifier
    model.classifier[3] = nn.Linear(4096, 512, bias=True)
    model.classifier[6] = nn.Linear(512, 2, bias=True)

    return model


def alexnet(input_layer=2):
    model = models.alexnet(pretrained=True)
    # # freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # change a few layers in classifier
    model.classifier[3] = nn.Linear(4096, 2048, bias=True)
    model.classifier[4] = nn.Linear(2048, 512, bias=True)
    model.classifier[6] = nn.Linear(512, 2, bias=True)

    return model


def resnet18_5(output_layer=5):
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

# model = vgg()
# model = alexnet()
# print(model)
