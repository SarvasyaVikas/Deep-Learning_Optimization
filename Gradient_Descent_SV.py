import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import argparse

def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=10000)
ap.add_argument("-s", "--samples", type=float, default=1000)
ap.add_argument("-a", "--alpha", type=float, default=0.01)
ap.add_argument("-n", "--notif", type=int, default=1)
args = vars(ap.parse_args())

(X, y) = make_blobs(n_samples=args["samples"], n_features=2, centers=2, cluster_std=1.05, random_state=20)

X = np.c_[np.ones((X.shape[0])), X]

W = np.random.uniform(size=(X.shape[1],))

lossHistory = []

for epoch in np.arange(0, args["epochs"]):
	preds = sigmoid_activation(X.dot(W))

	error = preds - y

	loss = np.sum(error ** 2)
	lossHistory.append(loss)
	if (epoch + 1) % args["notif"] == 0:
		print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))

	gradient = X.T.dot(error) / X.shape[0]

	W += -args["alpha"] * gradient
	
	if epoch > 100:
	
		loss_change = lossHistory[-2] - lossHistory[-1]
		change = args["samples"] / 50000
		
		if loss_change < change:
			break

Y = (-W[0] - (W[1] * X)) / W[2]

plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")

fig = plt.figure()
plt.plot(np.arange(0, len(lossHistory)), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
