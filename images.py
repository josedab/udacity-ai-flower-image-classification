import torch
from torchvision import datasets, transforms
from PIL import Image

imagenetMeans = [0.485, 0.456, 0.406]
imagenetStdDeviations = [0.229, 0.224, 0.225]

transformations = {
    'training':transforms.Compose([transforms.RandomRotation(30),
                                  transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(imagenetMeans, imagenetStdDeviations)]),
    'validation':transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(imagenetMeans, imagenetStdDeviations)]),
    'testing':transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(imagenetMeans, imagenetStdDeviations)])
}

def get_transformation(type):
    return transformations[type]

def get_dataset(data_dir, type):
    if(type == 'training'):
        dir = data_dir + '/train'
    elif(type == 'validation'):
        dir = data_dir + '/valid'
    elif(type == 'test'):
        dir = data_dir + '/test'
    else:
        raise Exception("Invalid Dataset type. Please specify training, validation or test")

    return datasets.ImageFolder(dir, transform=get_transformation(type))

def get_dataloader(dataset, batch_size=32, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_pil = Image.open(image)
    pre_processing_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenetMeans, std=imagenetStdDeviations)])

    return pre_processing_transforms(image_pil).numpy()