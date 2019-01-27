import argparse

def get_input_args_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'flowers',
                        help = 'Set directory to save checkpoints')

    parser.add_argument('--arch', type = str, default = 'vgg16',
                        help = 'Neural Network Model architecture to be used')
    parser.add_argument('--gpu', action= 'store_true' ,
                        help = 'Use GPU for training if available')
    parser.add_argument('--save_dir', type = str, default = './',
                        help = 'Set directory to save checkpoints')

    # Hyperparameters
    parser.add_argument('--learning_rate', type = float , default = 0.01,
                        help = 'Learning rate hyperparameter')
    parser.add_argument('--hidden_units', type = int , default = 512,
                        help = 'Hidden units hyperparameter')
    return parser.parse_args()


def get_input_args_prediction():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type = str,
                        help = 'Image to classify')
    parser.add_argument('--checkpoint', type = str,
                        help = 'Model to used when predicting the category image')

    # Optional
    parser.add_argument('--topk', type = int, default=5,
                        help='Return top K category predictions')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='JSON file with mappings between index classes and labels')
    parser.add_argument('--gpu', action= 'store_true',
                        help = 'Use GPU for training if available')

    return parser.parse_args()
