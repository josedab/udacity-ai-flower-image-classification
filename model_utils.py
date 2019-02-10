from torch import nn, optim
from torchvision import models
from collections import OrderedDict


# List of neural network architectures supported
architectures ={
    'vgg16': {
        'input': 25088
    },
    'alexnet': {
        'input': 9216
    },
    'densenet121': {
        'input': 1024
    },
    'inception_v3': {
        'input': 2048
    },
}

def load_pytorch_model(architecture):
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    else:
        raise Exception("Neural network architecture: {} not supported".format(architecture))
    return model

def create_classifier_to_train(architecture, hidden_layer_units, categories_to_classify):
    neural_network_architecture_inputs = architectures[architecture]['input']
    classifier = nn.Sequential(OrderedDict([
                                ('dropout',nn.Dropout(0.5)),
                                ('fc1', nn.Linear(neural_network_architecture_inputs, hidden_layer_units)),
                                ('relu1', nn.ReLU()),
                                ('fc2', nn.Linear(hidden_layer_units, 250)),
                                ('relu2', nn.ReLU()),
                                ('fc3', nn.Linear(250, categories_to_classify)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    return classifier

def create_transfer_learning_model(architecture='vgg16', hidden_layer_units=500, categories_to_classify=102):
    """
    Creates a neural network from a Pytorch supported architecture.
    Returns a neural network whose classifier needs to be trained.
    """
    # Load neural network architecture that is already trained and exposed in Pytorch
    model = load_pytorch_model(architecture)

    # No calculation of gradient for the pretrained neural network
    for param in model.parameters():
        param.requires_grad = False

    # Update classifier of the model with our own neural network
    model.classifier = create_classifier_to_train(architecture, hidden_layer_units, categories_to_classify)
    return model

def neural_network_setup(architecture, hidden_layer_units=500, learning_rate=0.01, categories_to_classify=102):
    model = create_transfer_learning_model(architecture, hidden_layer_units, categories_to_classify)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return {
        'architecture': architecture,
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer
    }
