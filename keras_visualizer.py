# KERAS visualiztion

import numpy as np
import librosa
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from vis.backprop_modifiers import guided

s_rate = 20000
seconds= 4
vec_len = s_rate * seconds

K.set_image_data_format("channels_first")

# load json and create model
json_file = open('./keras_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./keras_weights.h5")
print("Loaded model from disk")

model.summary()
model = guided(model)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_list = ['conv1d_2','conv1d_3','conv1d_4','conv1d_5']
nb_filters_list = [32,64,128,256]

input_img = model.input

def normalize_vector(v):
    return (v - min(v))/(max(v) - min(v))

def normalize_tensor(x,norm_param=1e-9):
  # Normalize a tensor by its l2 norm
  return x / (K.sqrt(K.mean(K.square(x))) + norm_param)

def visualize_layer(layer_idx, max_restarts=120, epochs = 100, step = 1.):
  layer_name = layer_list[layer_idx]
  nb_filters = nb_filters_list[layer_idx]

  samples = []
  for filter_index in range(nb_filters):
    print('Processing filter %d' % filter_index)
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:,filter_index,:])
    grads = K.gradients(loss, input_img)[0]
    grads = normalize_tensor(grads)
    iterate = K.function([input_img, K.learning_phase()],[loss,grads])
    for tries in range(max_restarts):
      again = False
      # Generate white noise from ~Uni[-0.015, 0.015)
      input_img_data = np.random.random((1,1,vec_len))
      input_img_data = (input_img_data - 0.5) * 0.03
          
      for i in range(epochs):
        loss_value, grads_value = iterate([input_img_data, 0]) # 0 test phase
        input_img_data += grads_value * step
        print('Current loss value:', loss_value)
        if loss_value <= 0.05 and i >= 3:
        # Bad starting point for search. try again
          again = True
          print("again..")
          break
      
      if not again:
        break
    if again:
      print('skipping', filter_index )
      continue

    sample = np.squeeze(input_img_data)
    samples.append(sample)
  return samples


def plot_frequencies(samples, fftsize=729):
  freqs = []
  centroids = []
  for sample in samples:
    spectral_centroids = librosa.feature.spectral_centroid(sample, sr=s_rate).squeeze()
    centroids.append(np.mean(spectral_centroids))
    
    # perform squared magnitude spectra
    S = librosa.core.stft(sample,n_fft=fftsize,hop_length=fftsize,win_length=fftsize)
    spec = librosa.core.amplitude_to_db(np.absolute(S))
    # Squeeze time bins to 1
    filter_freqs = np.mean(spec, axis = 1)
    filter_freqs = normalize_vector(filter_freqs)
    freqs.append(filter_freqs)

  freqs = np.array(freqs)
  argmaxed = np.argmax(freqs,axis=1)
  sort_idx = np.argsort(argmaxed)
  sorted_fft = freqs[sort_idx,:]

  plt.plot(np.argmax(sorted_fft, axis=1), ls='--', color='red')
  plt.pcolormesh(sorted_fft.T)
  plt.xlabel('Filters')
  plt.ylabel('Frequencies')
  plt.xticks([], "")
  plt.yticks([], "")
  plt.colorbar()
  plt.savefig('Frequencies.png')
  plt.close()
  
  plt.plot(np.sort(centroids))
  plt.xlabel("Filters")
  plt.ylabel("Central Frequencies [Hz]")
  plt.title("Central Frequency responses")
  plt.ylim(4200,6200)
  plt.savefig('Centroids.png')
  plt.close()

def plot_nb_onsets(samples, layer_idx):
  onsets = []
  for sample in samples:
    y = librosa.onset.onset_detect(y=sample, sr = s_rate)
    onsets.append(len(y))
  bins = range(0,45, 5)
  hist = [sum([1 for x in onsets if x >= start and x < start + 5]) for start in bins]
  hist = np.array(hist) / sum(hist)

  plt.plot(bins, hist, label="Layer nb. " + str(layer_idx))
  plt.xlabel("Number of onsets detected")
  plt.ylabel("Percentage of filters")
  plt.ylim(0, 0.5)
  plt.legend(loc='upper left')
  plt.title("Incidence of onsets")
  plt.savefig('Onsets.png')
