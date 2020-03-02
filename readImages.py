import gzip
import matplotlib.pyplot as plt
import numpy as np
f = gzip.open('C:\\Users\\Sid\\Documents\\Codes\\KMeansClustering\\train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 100

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

output = open("output.txt", "w")
image = np.asarray(data[1]).squeeze()
for line in image:
    output.write(str(line).replace("\n", ""))
    output.write("\n")

plt.imshow(image)
plt.show()