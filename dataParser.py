import gzip
# import matplotlib.pyplot as plt
import numpy as np

def readData(filepath, isTrain=True):
    file = gzip.open(filepath, 'r')
    image_size = 28

    if isTrain:
        num_images = 60000
    else:
        num_images = 10000

    file.read(16)
    buffer = file.read(image_size * image_size * num_images)
    data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size * image_size)
    
    # print(data.shape)
    # image = data[0].reshape(image_size, image_size)
    # plt.imshow(image)
    # plt.show()
    
    return data

