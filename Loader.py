import librosa
import os
import scipy.signal as signal
import numpy as np
import torch
import torch.utils.data as utils
import pickle


s_rate = 20000
seconds = 4
classes = []


def split_data(data, validation_split=0.1, batch_size=100):
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    random_seed = 37
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = utils.DataLoader(data, batch_size=batch_size,
                                    sampler=utils.SubsetRandomSampler(train_indices))
    validation_loader = utils.DataLoader(data, batch_size=1,
                                         sampler=utils.SubsetRandomSampler(val_indices))
    return train_loader, validation_loader


def highpass_filter(data, order=5):
    normal_cutoff = 700 / (0.5 * s_rate)
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    y = signal.sosfilt(sos, data)
    return y


def create_sample(audio_vector):
    max_offset = s_rate * seconds - len(audio_vector)
    if max_offset > 0:
        # need to pad recording
        offset = np.random.randint(max_offset)
        return np.pad(audio_vector, (offset, max_offset - offset), "constant")
    else:
        # need to limit recording
        filtered_vector = highpass_filter(audio_vector)
        max_index = int(np.argmax(filtered_vector))
        side_margin = int(s_rate * seconds / 2)
        max_index = min(max(max_index, side_margin), len(audio_vector) - side_margin)
        return audio_vector[max_index - side_margin: max_index + side_margin]


def load_data(from_pickle=False, spectrogram=False, batch_size = 100):
    if from_pickle:
        data = pickle.load(open("save.p", "rb"))
    else:
        global classes
        data = []
        for label, species in enumerate(os.listdir('files'), 0):
            classes.append(species)
            for recording in os.listdir('files/' + species):
                path = 'files/' + species + '/' + recording
                try:
                    audio_vector, sr = librosa.load(path, sr=s_rate)
                    sample = create_sample(audio_vector)
                    if spectrogram:
                        sample = librosa.feature.melspectrogram(sample, sr=s_rate, hop_length=256).astype('float32')
                        sample = np.stack((sample,) * 3)
                    else:
                        sample = np.expand_dims(sample, 0)
                    sample = torch.from_numpy(sample)
                except:
                    continue
                data.append((sample, label))
        pickle.dump(data, open("save.p", "wb"))
    return split_data(data, batch_size)
