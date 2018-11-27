from keras.models import load_model
import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import subprocess as sp


model = load_model("model.h5")
model.summary()
print("Welcome to the Digit tester .\n")
while (1):
	filepath=input("Drag a image from the test folder to test it .\n")
	image = misc.imread(filepath[54:-2])
	# plt.imshow(image)
	# plt.show()
	# predict results
	result = model.predict(image.reshape(-1,28,28,1))
	tmp = sp.call('clear',shell=True)
	# select the indix with the maximum probability
	result = np.argmax(result,axis = 1)
	print("The number must be : ",result)
