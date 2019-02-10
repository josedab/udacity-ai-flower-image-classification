import torch
from torch import nn
from torch import optim

def training_validation(model, dataloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for inputs, labels in dataloader:

        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def train_model(training_set, validation_set, model, optimizer, criterion, epochs = 5, print_every = 5, device='cuda'):

    model.to(device)

    steps = 0
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in training_set:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Evaluation mode for the validation
                model.eval()

                with torch.no_grad():
                    validation_loss, validation_accuracy = training_validation(model, validation_set, criterion, device)

                print("Epoch: {}/{}--- ".format(e+1, epochs),
                      "Training Loss: {:.4f}  ".format(running_loss/print_every),
                      "Validation Loss: {:.4f}  ".format(validation_loss/len(validation_set)),
                      "Validation Accuracy: {:.4f}  ".format(validation_accuracy/len(validation_set)))

                running_loss = 0

                # Back on training mode
                model.train()
