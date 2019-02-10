import torch
from images import process_image


def predict(image_path, model, topk=5, is_gpu_enabled=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.eval()

    if is_gpu_enabled:
        model = model.cuda()
    else:
        model = model.cpu()

    # TODO: Implement the code to predict the class from an image file
    flower_img = process_image(image_path)
    flower_tensor = torch.from_numpy(flower_img)

    if is_gpu_enabled:
        flower_tensor = flower_tensor.cuda()

    flower_tensor.unsqueeze_(0)

    with torch.no_grad():
        output = model.forward(flower_tensor)

    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}

    softmax_probability_labels = torch.exp(output).topk(topk)

    probabilities = softmax_probability_labels[0][0]
    labels = softmax_probability_labels[1][0]

    # Move back probs and labels if on gpu
    if is_gpu_enabled:
        probabilities = probabilities.cpu()
        labels = labels.cpu()

    classes = [class_to_idx_inverted[label] for label in labels.numpy()]

    return probabilities.numpy(), classes
