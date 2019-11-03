# gzip library is used for reading the MNIST dataset
import gzip
import numpy as np


def readData(filepath, isTrain=True):
    """readData reads the MNIST image data in ubyte format and converts
       it into an array of type N x 784, where N is the no of samples
       Parameter:
            filepath: path to test/ training image gz file
            isTrain: boolean to choose between test/ training data
                     No of images count is taken as 60,000 for training
                     data and 10,000 for test data
       Returns:
            image data as array of type (No. of samples x 784)
    """
    # Define image size as 28x28
    image_size = 28
    # Open the gz file
    file = gzip.open(filepath, 'r')
    
    # If trainging data selected, set no. of images to 60,000 else 
    # set it to 10,000
    if isTrain:
        num_images = 60000
    else:
        num_images = 10000
    
    # Read out the file header (16 bytes)
    file.read(16)
    # Read the entire file and reshape to 
    # (no of images, image size * image size)
    buffer = file.read(image_size * image_size * num_images)
    data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size * image_size)
    # Convert image from 8b to binary
    data[data < 100] = 0
    data[data >= 100] = 1
    #Close the file
    file.close()
    
    # Return the image data as array
    return data

def readDataset(data_filepath, label_filepath, isTrain=True):
    """readDataset reads the MNIST training/ test dataset (images and labels)
       and returns the data and label (one hot encoded)
       Parameter:
            data_filepath: path to test/ training image gz file
            label_filepath: path to test/ training label gz file
            isTrain: boolean to choose between test/ training data
                     No of images count is taken as 60,000 for training
                     data and 10,000 for test data
       Returns:
            image data as array of type (No. of samples x 784) and label data
            as one hot encoded vector array of type (No. of samples x 10)
    """
    # Define image size as 28x28
    image_size = 28
    # If trainging data selected, set no. of images to 60,000 else 
    # set it to 10,000
    if isTrain:
        num_images = 60000
    else:
        num_images = 10000

    # Open the image gz file
    file = gzip.open(data_filepath, 'r')
    # Read out the file header (16 bytes)
    file.read(16)
    # Read the entire file and reshape to 
    # (no of images, image size * image size)
    buffer = file.read(image_size * image_size * num_images)
    data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size * image_size)
    # Convert image from 8b to binary
    data[data < 100] = 0
    data[data >= 100] = 1
    #Close the file
    file.close()

    # Open the label gz file
    file = gzip.open(label_filepath, 'r')
    # Read out the file header (8 bytes)
    file.read(8)
    # Read the entire file and reshape to (no of images, 1)
    buffer = file.read(num_images)
    label = np.frombuffer(buffer, dtype=np.uint8).astype(np.int)
    label = label.reshape(num_images, 1)
    label = np.array(label)
    label = np.concatenate(label, axis=0)
    # Convert labels from 8b integers to one hot encoded vectors of size 10
    label_one_hot = np.zeros((num_images, 10), dtype=int)
    label_one_hot[np.arange(num_images), label] = 1
    #Close the file
    file.close()

    # Return the image data as array and one hot encoded label data
    return [data, label_one_hot]
