import torch

from get_input_args import get_input_args_training
from checkpoint_utils import save_model
from training_utils import train_model

from images import get_dataset

from model_utils import neural_network_setup

def main():
    arguments = get_input_args_training()

    device = "cpu"
    is_gpu_enabled =  torch.cuda.is_available() and arguments.gpu
    if is_gpu_enabled:
        device = "cuda"

    print("Transfer learning process starting")
    print("- Loading dataset")
    data_dir = arguments.data_dir
    training_set = get_dataset(data_dir, 'training')
    validation_set = get_dataset(data_dir, 'validation')

    architecture = arguments.arch
    hidden_layer_units = arguments.hidden_units
    learning_rate = arguments.learning_rate
    flower_categories_cardinality = 102
    print(arguments.arch)

    print("- Creating transfer learning model")
    neural_network = neural_network_setup(architecture, hidden_layer_units, learning_rate, flower_categories_cardinality)
    model = neural_network['model']
    criterion = neural_network['criterion']
    optimizer = neural_network['optimizer']

    print("- Training model")
    train_model(training_set, validation_set, model, optimizer, criterion, epochs = 5, print_every = 5, device=device)

    print("- Saving model into disk")
    model_file_path = arguments.save_dir + "/model"
    save_model(model_file_path, architecture, model, hidden_layer_units)

    print("Training process completed.")




if __name__ == "__main__":
    main()
