import numpy as np

#Normalizes matrix values to values between -1 and 1
def normalize(data):
	#data: training examples x features
	for feature in data.T:
		mean = np.mean(feature)
		range = np.max(feature) - np.min(feature)
		#print(mean, range)
		feature -= mean
		feature /= range
	return data

#arr = np.array([[1000.0, 32.0],[2000.0,4.0], [10000.0, 37.0]])
#print(normalize(arr))

def predict(data, params):
	#data: data predictions x features
	#params: features x 1
	return np.dot(data, params)  
