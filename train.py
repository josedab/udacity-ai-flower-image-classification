import torch

from get_input_args import get_input_args_training
from checkpoint_utils import save_model
from training_utils import train_model
from images import get_dataloader, get_dataset
from model_utils import neural_network_setup


def main():
    arguments = get_input_args_training()

    device = "cpu"
    is_gpu_enabled = torch.cuda.is_available() and arguments.gpu
    if is_gpu_enabled:
        device = "cuda"
        print("Training model using GPU")
    else:
        print("Training model without using GPU")

    print("Transfer learning process starting")
    print("- Loading dataset")
    data_dir = arguments.data_dir
    training_dataset = get_dataset(data_dir, 'training')
    validation_dataset = get_dataset(data_dir, 'validation')
    training_data_loader = get_dataloader(training_dataset)
    validation_data_loader = get_dataloader(validation_dataset)

    architecture = arguments.arch
    hidden_layer_units = arguments.hidden_units
    learning_rate = arguments.learning_rate
    flower_categories_cardinality = 102
    epochs = arguments.epochs
    print(arguments.arch)

    print("- Creating transfer learning model")
    neural_network = neural_network_setup(architecture, hidden_layer_units, learning_rate,
                                          flower_categories_cardinality)
    model = neural_network['model']
    model.class_to_idx = training_dataset.class_to_idx

    criterion = neural_network['criterion']
    optimizer = neural_network['optimizer']

    print("- Training model")
    train_model(training_data_loader, validation_data_loader, model, optimizer, criterion, epochs=epochs, print_every=10, device=device)

    print("- Saving model into disk")
    model_file_path = arguments.save_dir + "checkpoint.pth"
    save_model(model_file_path, architecture, model, hidden_layer_units)
    print("- Model saved at {}".format(model_file_path))

    print("Training process completed.")


if __name__ == "__main__":
    main()
