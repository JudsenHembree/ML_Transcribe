""" Here we define the model architecture and training loop """
import torch
import os
from torch import nn
from torch.nn import init

class Model(nn.Module):
    def __init__(self, number_of_layers=4):
        super().__init__()
        conv_layers = []
        self.number_of_layers = number_of_layers

        starting_channels = 2 # 2 channels for the stereo image
        # build up to a peak number of nodes
        for i in range(self.number_of_layers):
            conv_layers += [nn.Conv2d(starting_channels, 2**(i+3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))]
            conv_layers += [nn.ReLU()]
            conv_layers += [nn.BatchNorm2d(2**(i+3))]
            init.kaiming_normal_(conv_layers[-3].weight, a=0.1)
            conv_layers[-3].bias.data.zero_()
            starting_channels = 2**(i+3)
        # then build down
        for i in range(self.number_of_layers, 2, 1):
            conv_layers += [nn.Conv2d(starting_channels, 2**(i+2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))]
            conv_layers += [nn.ReLU()]
            conv_layers += [nn.BatchNorm2d(2**(i+2))]
            init.kaiming_normal_(conv_layers[-3].weight, a=0.1)
            conv_layers[-3].bias.data.zero_()
            starting_channels = 2**(i+2)
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=starting_channels, out_features=3)

        self.conv = nn.Sequential(*conv_layers)
 
    def forward(self, x):
        x = self.conv(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        x = self.lin(x)

        return x


def training(model, train_dl, num_epochs, device, log = False, log_dest = 'log.csv'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
    f = None
    if log:
        f = open(log_dest, 'w')
        f.write('Epoch, Loss, Accuracy\n')
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs,1)
            print("Expecting: " + str(labels) + " Got: " + str(prediction))
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        if log:
            if f is not None:
                f.write(f'{epoch}, {avg_loss}, {acc}\n')

    if f is not None:
        f.close()
    print('Finished Training')
      
def save_model(model, model_path):
    """ Save the model to the given path """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, 'model.pt')
    torch.save(model.state_dict(), model_path)
