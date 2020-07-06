import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.models as models
import Loader
import numpy as np


saved_state_path = 'torch_trained_weights'
num_classes = 10

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = None


class TweetNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, 64, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(8, stride=8)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, 32, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(8, stride=8)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, 16, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, 8, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(4, stride=4)
        )

        self.feature_extractor = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
            nn.Softmax(1)
        )

    def extract_features(self, x):
        for layer in self.feature_extractor:
            x = layer(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)


def train(trainloader, epochs = 100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters())

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = net(inputs)
            outputs = outputs.to(device)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            # backward + optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("loss for epoch", epoch, running_loss / len(trainloader))
        if epoch % 10 == 0:
            torch.save(net.state_dict(), saved_state_path)

    print('Finished Training')
    torch.save(net.state_dict(), saved_state_path)


def test_net(validation_set):
    # Load net parameters
    net.load_state_dict(torch.load(saved_state_path))
    net.eval()

    success_counters = np.zeros(num_classes)
    species_counters = np.zeros(num_classes)
    for instance in validation_set:
        input, label = instance
        species_counters[label] += 1

        input = input.to(device)
        res = net(input).cpu().detach()
        prediction = res.numpy().argmax()

        if prediction == label:
            success_counters[label] += 1

    print('success rate per species:')
    print([success / total for success, total in zip(success_counters, species_counters)])
    print('total success rate:' + str(sum(success_counters) / sum(species_counters)))


def main(spectrogram=False):
    global net
    if spectrogram:
        net = models.vgg19_bn(num_classes=num_classes)
        batch_size = 50
    else:
        net = TweetNet()
        batch_size = 100

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.to(device)

    train_loader, validation_loader = Loader.load_data(spectrogram=spectrogram, batch_size=batch_size)
    train(train_loader)
    test_net(validation_loader)


if __name__ == "__main__":
        main()
