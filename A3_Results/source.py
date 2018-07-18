import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn.utils as sklearn
import sklearn.linear_model as lin
import bonnerlib2	
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax
from sklearn.utils import gen_batches

########### QUESTION 1 #############

colors = np.array(['r','b'])

X_train, t_train = datasets.make_moons(n_samples=200, noise=0.2)
X_test, t_test = datasets.make_moons(n_samples=10000, noise=0.2)

fig1 = plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], color=colors[t_train],s=2)
fig1.suptitle("Figure 1, Question 1(a): Moons Training Data")

fig2 = plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], color=colors[t_test],s=2)
fig2.suptitle("Figure 2, Question 1(a): Moons Test Data")

def fitMoons():
	fig_contour = plt.figure()
	fig_contour.suptitle("Figure 3, Question 1(b): Contour plots for various training sessions")

	clf = MLPClassifier(hidden_layer_sizes=[3],
					activation='tanh',
					solver='sgd',
					learning_rate_init=0.01,
					tol=10.0**(-20),
					max_iter=10000)

	errTestList = []
	errTestMin = np.Inf
	errTestCLF = np.Inf
	for x in range(9):
		current_clf = clf.fit(X_train, t_train)
		testing_err = 1 - clf.score(X_test, t_test)
		errTestList.append(testing_err)
		print("Testing error for training session %s = %s" % (x+1, testing_err))
		ax = fig_contour.add_subplot(3, 3, x+1)
		ax.scatter(X_train[:, 0], X_train[:, 1], color=colors[t_train],s=2)
		bonnerlib2.dfContour(current_clf, ax)

		if testing_err < errTestMin:
			errTestMin = testing_err
			errTestCLF = current_clf
	
	fig_best = plt.figure()
	fig_best.suptitle("Figure 4, Question 1(b): Contour plot for best training session")		
	print("Smallest test error = %s" % (errTestMin))
	ax = fig_best.add_subplot(1, 1, 1)
	ax.scatter(X_train[:, 0], X_train[:, 1], color=colors[t_train],s=2)
	bonnerlib2.dfContour(errTestCLF, ax)

fitMoons()

###########	END OF QUESTION 1 ############

########### QUESTION 3 #############

with open('mnist.pickle','rb') as f:
	data = pickle.load(f)

def flatten(data):
	X = np.vstack(data)
	t = np.zeros(np.shape(X)[0], dtype='int')
	m1 = 0
	m2 = 0
	for i in range(0, len(data)):
		m = np.shape(data[i])[0]
		m2 = m2 + m
		t[m1:m2] = i
		m1 = m1 + m
	return X, t

# Question 3(a) and (b)
X, t = flatten(data['training'])
X_shuffled, t_shuffled = sklearn.shuffle(X, t)
X_test, t_test = flatten(data['testing'])

scalar = StandardScaler()
scalar.fit(X,t)
X_normal = scalar.transform(X_shuffled)
X_test_normal = scalar.transform(X_test)

def displaySample(N, D):
	array = sklearn.resample(D, replace='False', n_samples=N)
	fig_digit = plt.figure()
	for row in range(N):
		ax = fig_digit.add_subplot(np.ceil(np.sqrt(N)), np.ceil(np.sqrt(N)), row+1)
		image = np.reshape(array[row], (28, 28))
		im = plt.imshow(image, cmap='Greys', interpolation='nearest')
		plt.axis('off')
	return ax

# creates matrix of target values where
# each row is (t1, t2, ..., tk) 
# and t_k = 1 if x is in that class
t_test_prime = np.zeros(shape=(len(t_test), 10))
t_test_prime[np.arange(len(t_test)), t_test] = 1
t_test_matrix = np.array(t_test_prime.astype(int))

t_shuffled_prime = np.zeros(shape=(len(t_shuffled), 10))
t_shuffled_prime[np.arange(len(t_shuffled)), t_shuffled] = 1 
t_shuffled_matrix = np.array(t_shuffled_prime.astype(int))

# Question 3(c)
ax = displaySample(16, X_normal)
plt.suptitle("Question 3(c): some normalized MNIST digits")

# Question 3(d)
clf = MLPClassifier(hidden_layer_sizes=[100],
				activation='tanh',
				solver='sgd',
				#batch_size=60000,
				batch_size=200,
				tol=0.0,
				max_iter=5,
				warm_start=True,
				learning_rate_init=0.18,
				momentum=0.65,
				alpha=0.165)

trainErrList = []
testErrList = []
for x in range(50):
	clf.fit(X_normal, t_shuffled)
	testing_err = round((1 - clf.score(X_test_normal, t_test))*100, 2)
	training_err = round((1 - clf.score(X_normal, t_shuffled))*100, 2)
	trainErrList.append(training_err)
	testErrList.append(testing_err)
	print("Testing error for training session %s = %s" % (x+1, testing_err))
	print("Training error for training session %s = %s" % (x+1, training_err))

coefs = clf.coefs_
bias = clf.intercepts_
predict_proba = clf.predict_proba(X_test_normal)

# Question 3(f)
fig5 = plt.figure()	
fig5.suptitle("Figure 5, Question 3: training and test error in batch mode")
plt.plot(trainErrList, color='orange')
plt.plot(testErrList, color='blue')
plt.xlabel('training iterations')
plt.ylabel('error')

fig6 = plt.figure()
fig6.suptitle("Figure 6: Question 3: test error during last 500 iterations of batch training")
plt.plot(testErrList[-500:], color='blue')
plt.xlabel('training iterations')
plt.ylabel('error')

###########	END OF QUESTION 3 ############

########### QUESTION 4 #############

def tanh(x):
	return np.tanh(x)

def softmax(x):
	return np.exp(x) / np.exp(x).sum(keepdims=True, axis=1)

# Question 4(a)
def predict(X, W1, W2, b1, b2):
	hidden_matrix = tanh(np.dot(X, W1) + b1.reshape(1, -1))
	output_matrix = softmax(np.dot(hidden_matrix, W2) + b2.reshape(1, -1))
	return hidden_matrix, output_matrix

h1, output = predict(X_test_normal, coefs[0], coefs[1], bias[0], bias[1])

# Question 4(b)
print(np.sum((output - predict_proba)**2))

# Question 4(c)
def gradient(H, Y, T):	
	output_error = Y - T
	DW = (np.matmul(output_error.T, H)) / len(X_normal)
	Db = np.mean(output_error, 0)
	return DW.T, Db.reshape(-1,1)

# Question 4(d)
trainingErrList = []
testingErrList = []
meanLoss = []
def bgd(W1, b1, lrate, sigma, K):
	W2 = sigma * np.random.randn(100, 10) + 0
	b2 = np.zeros(shape = [10, 1])
	for i in range(K+1):
		hidden_grad, output_grad = predict(X_normal, W1, W2, b1, b2)
		test_hidden, test_output = predict(X_test_normal, W1, W2, b1, b2)
		dW2, db2 = gradient(hidden_grad, output_grad, t_shuffled_matrix)
		W2 = W2 - lrate*dW2
		b2 = b2 - lrate*db2
		if np.mod(i, 5) == 0:
			meanloss = round((np.mean(-t_shuffled_matrix * np.log(output_grad))*100), 2)
			testing_err = round((np.sum((t_test_matrix - test_output)**2)/len(t_test_matrix)*100), 2)
			training_err = round((np.sum((t_shuffled_matrix - output_grad)**2)/len(t_shuffled_matrix)*100), 2)
			meanLoss.append(meanloss)
			trainingErrList.append(training_err)
			testingErrList.append(testing_err)
			print("Iteration number = %s" %(i))
			print("Testing error for training session %s = %s" % (i, testing_err))
			print("Training error for training session %s = %s" % (i, training_err))
			print("Mean training loss for training session %s = %s" % (i, meanloss))

# Question 4(e)
bgd(coefs[0], bias[0], 0.18, 0.01, 1000)

# Question 4(d) graph outputs
fig7 = plt.figure()	
fig7.suptitle("Figure 7, Question 4(d): training and test error for batch gradient descent")
plt.plot(trainingErrList, color='orange')
plt.plot(testingErrList, color='blue')
plt.xlabel('iterations')
plt.ylabel('error')

fig8 = plt.figure()
fig8.suptitle("Figure 8: Question 4(d): mean training loss for batch gradient descent")
plt.plot(meanLoss, color='orange')
plt.xlabel('iterations')
plt.ylabel('loss')

fig9 = plt.figure()
fig9.suptitle("Figure 9: Question 4(d): training and test error for last 500 epochs of bgd")
plt.plot(trainingErrList[-100:], color='orange')
plt.plot(testingErrList[-100:], color='blue')
plt.xlabel('iterations')
plt.ylabel('error')

fig10 = plt.figure()
fig10.suptitle("Figure 10: Question 4(d): mean training loss for last 500 epochs of bgd")
plt.plot(meanLoss[-100:], color='orange')
plt.xlabel('iterations')
plt.ylabel('loss')

# Question 4(g)
stochtrainingErrList = []
stochtestingErrList = []
stochmeanLoss = []
def sgd(W1, b1, lrate, alpha, sigma, K, batchSize, mom):
	W2 = sigma * np.random.randn(100, 10) + 0
	b2 = np.zeros(shape = [10, 1])
	vdW2 = np.zeros(shape=[100, 10])
	vdb2 = np.zeros(shape=[10, 1])
	for i in range(K+1):
		for batch_slice in gen_batches(len(X_normal), batchSize):
			X_train_batch = X_normal[batch_slice]
			T_train_batch = t_shuffled_matrix[batch_slice]
			hidden_grad, output_grad = predict(X_train_batch, W1, W2, b1, b2)
			dW2, db2 = gradient(hidden_grad, output_grad, T_train_batch)
			dW2 = dW2 + alpha*W2
			vdW2 = mom*vdW2 + dW2
			vdb2 = mom*vdb2 + db2
			W2 = W2 - lrate*vdW2
			b2 = b2 - lrate*vdb2
		if np.mod(i, 5) == 0:
			test_hidden, test_output = predict(X_test_normal, W1, W2, b1, b2)
			hidden_grad, output_grad = predict(X_normal, W1, W2, b1, b2)
			meanloss = round((np.mean(-t_shuffled_matrix * np.log(output_grad))*100), 2)
			meanloss = round((meanloss + (0.5*alpha) * (np.mean(np.square(W2)))), 2)
			training_err = round((np.sum((t_shuffled_matrix - output_grad)**2)/len(t_shuffled_matrix)*100), 2)
			testing_err = round((np.sum((t_test_matrix - test_output)**2)/len(t_test_matrix)*100), 2)
			stochmeanLoss.append(meanloss)
			stochtrainingErrList.append(training_err)
			stochtestingErrList.append(testing_err)
			print("Iteration number = %s" %(i))
			print("Testing error for training session %s = %s" % (i, stochtestingErrList[-1]))
			print("Training error for training session %s = %s" % (i, stochtrainingErrList[-1]))
			print("Mean training loss for training session %s = %s" % (i, stochmeanLoss[-1]))
	print("Minimum test error during %s epochs of training = %s" %(K, min(stochtestingErrList)))

# Question 4(h)
sgd(coefs[0], bias[0], 0.18, 0.0001, 0.01, 50, 3000, 0.99)

# Question 4(g) graph outputs
fig11 = plt.figure()	
fig11.suptitle("Figure 11, Question 4(g): training and test error for stochastic gradient descent")
plt.plot(stochtrainingErrList, color='orange')
plt.plot(stochtestingErrList, color='blue')
plt.xlabel('iterations')
plt.ylabel('error')

fig12 = plt.figure()
fig12.suptitle("Figure 12: Question 4(g): mean training loss for stochastic gradient descent")
plt.plot(stochmeanLoss, color='orange')
plt.xlabel('iterations')
plt.ylabel('loss')

plt.show()