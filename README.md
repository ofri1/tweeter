Downloader.py - An interface for downloading the data from the xeno-canto database. using tkinter, a python-included
package. The function receives species names and numbers of pages to download.

Loader.py - The file includes the main function for the processing of the raw audio data.
This includes the highpass filtering, the sampling of the relevant parts and the split into training and validation groups.

TweetTrain.py - Main file for the project. includes the definition of the net; the training of the net and the testing function.

torch_trained_weights - the trained weights for the net, for future loading.

visualize_net.py - includes several functions for the visualization of the net presented in the report.

keras_visualizer.py - Includes the function for the visualization of the convolutional layers (excluding the first layer)
using the keras package, with a keras model of our net.

keras_model.json - definition of the keras model.

keras_weights.h5 - the trained weights for the keras model, for future loading.

Instructions to run project:
1. Download data with Downloader.py
2. Train the net and test it using TweetTrain, which will save the trained paramaters to torch_trained_weights (included in files).
3. Visualize the torch model using the functions from visualize_net.py 
4. Visualize convolutional layers with keras_visualizer.py which uses the included trained Keras model.
