from torchvision import transforms



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

datasets = {
    'training': datasets.ImageFolder(train_dir, transform=transformations['training']),
    'validation' = datasets.ImageFolder(valid_dir, transform=transformations['validation']),
    'testing' = datasets.ImageFolder(test_dir, transform=transformations['testing'])
}

def get_transformation(type):
    return transformations[type]

def get_dataset(data_dir, type):
    if(type == 'training'):
        dir = data_dir + '/train'
    elif(type == 'validation'):
        dir = data_dir + '/valid'
    elif(type=='test'):
        dir = data_dir + '/test'
    else
        raise Exception("Invalid Dataset type. Please specify training, validation or test")

    return datasets.ImageFolder(dir, transform=get_transformation(type))
