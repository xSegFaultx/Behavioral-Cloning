from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


#model = load_model('model.h5')
#model.summary()

def generate_graph(file_path):
    csv_file = pd.read_csv(file_path)
    steering = np.array(csv_file['steering'])
    x = np.array(
        [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.hist(steering, range=(-1,1), bins=20)
    plt.xticks(x)
    plt.title(file_path[:-4])
    #plt.savefig(file_path[:-4] + '.jpg')
    #plt.show()



generate_graph('keyboard.csv')
generate_graph('joystick.csv')

image = cv2.imread(pd.read_csv('driving_log.csv')['center'][3])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
h, w = image.shape[:2]
[x1, x2] = np.random.choice(w, 2, replace=False)
k = h / (x2 - x1)
b = - k * x1
for i in range(h):
    c = int((i - b) / k)
    image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
plt.imshow(image)
plt.show()