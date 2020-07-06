import TweetTrain
import Loader
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_CL1():
    net = TweetTrain.TweetNet()
    net.load_state_dict(torch.load(TweetTrain.saved_state_path, map_location=torch.device('cpu')))
    net.eval()
    first_layer_weights = net.layer1[0].weight.detach()
    for i, feature_map in enumerate(first_layer_weights, 1):
        feature_vector = feature_map[0].tolist()

        plt.plot(feature_vector)
        plt.xlabel('Time (1 / Sampling rate)')
        plt.title('Feature map nb. ' + str(i))
        plt.savefig('cl1_test/raw' + str(i) + '.png')
        plt.close()

        x, y = signal.freqz(feature_vector, fs=Loader.s_rate)
        plt.plot(x, abs(y))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Frequency response [dB]')
        plt.title('Frequency responses for feature map nb. ' + str(i))
        plt.savefig('cl1_test/freq' + str(i) + '.png')
        plt.close()


class test_fc_layers(layer):
    def hook(module,input,output):
	    self.res = output

    def __init__(self):
        global device
        self.res = []
        self.net = TweetTrain.TweetNet()
        self.net.to(device)
        self.net.load_state_dict(torch.load(saved_state_path))
        self.net.eval()
        self.net.classifier[layer].register_forward_hook(hook)

    def visualize(self):
        global device
        arr = []
        colors = []

        train_loader, validation_loader = Loader.load_data(from_pickle=True, batch_size=1) 

        for x in train_loader:
            input, label = x
            input = input.to(device)
            self.net(input)
            res = self.res.cpu().detach().numpy()[0]
            arr.append(res)
            colors.append(label.detach().numpy()[0])

        X = (TSNE(n_components=2).fit_transform(arr))

        x = [r[0] for r in X]
        y = [r[1] for r in X]

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=colors)
        legend = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
        ax.add_artist(legend)

        plt.savefig('TSNE_test'+str(layer)+'.png')


class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.clone()

    def close(self):
        self.hook.remove()


class FilterVisualizer:
    def __init__(self, size):
        self.size = size
        self.model = TweetNet()
        self.model.load_state_dict(torch.load(saved_state_path))
        for layer in self.model.feature_extractor:
            for inner_layer in layer:
                inner_layer.trainable = False
        if size != Loader.s_rate * Loader.seconds:
            self.model.forward = self.model.extract_features
        self.model.to(device).eval()

    def visualize(self, layer_idx, filter_idx, initial_freq=None, opt_steps=50000):
        assert layer_idx > 0 # To fit enumertion in Tweetnet, count from 1
        layer = self.model.feature_extractor[layer_idx - 1][1]
        assert isinstance(layer, torch.nn.ReLU)
        # register hook.
        activations = SaveFeatures(layer)
        if initial_freq is not None:
            # generate sine wave
            vec = np.sin(2 * np.pi * initial_freq * (np.arange(self.size) / float(s_rate)))
            vec = torch.tensor(np.reshape(vec, (1, 1, -1)), dtype=torch.float32, requires_grad=True, device=device)
        else:
            # generate random initial vector
            vec = torch.randn((1, 1, self.size), dtype=torch.float32, requires_grad=True, device=device)
        optimizer = torch.optim.Adadelta([vec])

        for n in range(opt_steps):  # optimize vector entries for opt_steps times
            optimizer.zero_grad()
            self.model(vec)
            loss = -activations.features[0, filter_idx].mean()
            loss.backward()
            optimizer.step()

        # Remove hook
        activations.close()
        return vec.cpu().detach().numpy().squeeze()

    def show_frequencies(self, layer_idx, filter_idx):
        vec = self.visualize(layer_idx, filter_idx)
        return np.mean(np.abs(librosa.core.stft(res, n_fft=1024)), axis=1)

    def plot_frequencies(self, layer_idx):
        nb_filters_by_layer = [16, 32, 64, 128, 256]
        nb_filters = nb_filters_by_layer[layer_idx - 1]
        freqs_per_filter = [self.show_frequencies(layer_idx, filter_idx) for filter_idx in range(nb_filters)]
        freqs_per_filter.sort(key=lambda x: np.argmax(x))
        plt.plot(nb_filters, [np.argmax(x) for x in freqs_per_filter], ls='--', color='red')
        plt.pcolormesh(np.transpose(np.array(freqs_per_filter)))
        plt.ylabel('Frequency bins')
        plt.xlabel('Filters')
        plt.show()
