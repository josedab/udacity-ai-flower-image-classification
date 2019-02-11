import torch
from get_input_args import get_input_args_prediction
from checkpoint_utils import load_model
from prediction_utils import predict
from classes_utils import get_category_mappings


def main():
    arguments = get_input_args_prediction()
    print(arguments)

    is_gpu_enabled = torch.cuda.is_available() and arguments.gpu
    if is_gpu_enabled:
        device = "cuda"

    model_file_path = arguments.checkpoint
    model = load_model(model_file_path)

    probabilities, labels = predict(image_path=arguments.input,
                                    model=model,
                                    topk=arguments.topk,
                                    is_gpu_enabled=is_gpu_enabled)

    category_mappings_path = arguments.category_names
    if len(category_mappings_path) > 0:
        category_mappings = get_category_mappings(category_mappings_path)
        category_labels = [category_mappings[str(folder_idx)] for folder_idx in labels]
        labels = category_labels

    print_prediction_results(labels, probabilities, arguments.topk)


def print_prediction_results(labels, probabilities, topk=5):
    print("==================================")
    print("Prediction results")
    print("==================================")

    for label, probability in list(zip(labels, probabilities))[:topk]:
        print("Class: {}, Probability: {:.4f}".format(label, probability))


if __name__ == "__main__":
    main()
