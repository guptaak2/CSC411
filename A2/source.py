import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn.utils as sklearn
import sklearn.linear_model as lin
import bonnerlib2	

# function to generate random data based on mu0 and mu1 and N = sample size

def genData(mu0, mu1, N):
	# initialize 2 arrays for class 0 and class 1
	class0 = np.array((map(lambda x: x + mu0, (np.random.randn(N, 2)))))
	class1 = np.array((map(lambda x: x + mu1, (np.random.randn(N, 2)))))
	X = np.concatenate((class0, class1), axis=0)
	X = sklearn.shuffle(X, random_state=(np.random.randint(0, 50)))
	t = np.empty([2*N], dtype=int)
	for i in range(len(t)):
		if X[i] in class0:
			t[i] = 0
		else:
			t[i] = 1
	return X, t

# set values to call genData with
mu0 = (0.5, -0.5)
mu1 = (-0.5, 0.5)
N = 10000
betaList = [0.1,0.2,0.5,1.0]

# import data
with open('mnist.pickle','rb') as f:
	data = pickle.load(f)

# use logistic regression on generated data from genData
X, t = genData(mu0, mu1, N)
clf = lin.LogisticRegression()
clf.fit(X, t)
colors = np.array(['r','b'])

fig1 = plt.figure()
plt.scatter(X[:, 0], X[:, 1], color=colors[t],s=10)
fig1.suptitle("Figure 1: scatter plot of data")

# function to plot contour plot and surface plot of logistic decision function
def logregDemo(N, betaList):
	mu0 = (2, -2)
	mu1 = (-2, 2)
	m = len(betaList)
	fig_contour = plt.figure()
	fig_3d = plt.figure()
	fig_contour.suptitle("Figure 2: contour plot of logistic decision function")
	fig_3d.suptitle("Figure 3: surface plot of logistic decision function")
	# for each item in betaList
	for b_k in range(m):
		X, t = genData(betaList[b_k] * np.array(mu0), b_k * np.array(mu1), N)
		clf = lin.LogisticRegression()
		clf.fit(X, t)
		ax = fig_contour.add_subplot(np.ceil(m/2), np.ceil(m/2), b_k+1)
		ax2 = fig_3d.add_subplot(np.ceil(m/2), np.ceil(m/2), b_k+1, projection='3d')
		ax.set_xlim(-6, 6)
		ax.set_ylim(-6, 6)
		ax2.set_xlim(-9, 6)
		ax2.set_ylim(-6, 9)
		ax.scatter(X[:, 0], X[:, 1], color=colors[t],s=0.1)
		bonnerlib2.dfContour(clf,ax)
		bonnerlib2.df3D(clf,ax2)

logregDemo(N, betaList)

def displaySample(N, D):
	array = sklearn.resample(D, replace='False', n_samples=N)
	fig_digit = plt.figure()
	for row in range(N):
		ax = fig_digit.add_subplot(np.ceil(np.sqrt(N)), np.ceil(np.sqrt(N)), row+1)
		image = np.reshape(array[row], (28, 28))
		im = plt.imshow(image, cmap='Greys', interpolation='nearest')
		plt.axis('off')
	return ax

# function to flatten training/testing data
# loops through each class and appends to array X
# whilst adding class name to array t
def flatten(data):
	X = []
	t = []
	for i in range(len(data)):
		X.append(data[i])
		for digit in range((data[i].shape[0])):
			t.append(i)
	return np.vstack(X), np.array(t)

D = data['training'][5]
ax = displaySample(15, D)
plt.suptitle("Figure 4: random MNIST images of the digit 5")

D_2 = flatten(data['training'])[0]
ax_2 = displaySample(23, D_2)
plt.suptitle("Figure 5: random sample of MNIST training images")

max_probs = [] # to hold maximum probabilities to get least confident images
lowest_prob = [] # to hold images with least confident predicition
X_difference = [] # to hold misclassified images

# multi-class logistic regression to classify digits from training data
X, t = flatten(data['training'])
X_test, t_test = flatten(data['testing'])
clf = lin.LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(X, t)
training_err = clf.score(X, t) # 0.934066666667
testing_err = clf.score(X_test, t_test) # 0.9252
predict_train = clf.predict(X)
predict_test = clf.predict(X_test)
predict_probs = clf.predict_proba(X_test)

# for loop to find missclassfied images by checking 
# if test probabilites match predicted probabilities

for c in range(t_test.shape[0]):
	if t_test[c] != predict_test[c]:
		X_difference.append(X_test[c])

for prob in range(predict_probs.shape[0]):
	max_probs.append(np.max(predict_probs[prob]))

# find images with least confident predictions
least_confidence = np.argsort(max_probs)
least_confidence = least_confidence[:36]

for index in least_confidence:
	lowest_prob.append(X_test[index])

ax_3 = displaySample(36, X_difference)
plt.suptitle("Figure 6: some missclassified images")

ax_4 = displaySample(36, lowest_prob)
plt.suptitle("Figure 7: images with the least confident predictions")

def question_4(data):
	# flatten training[2] and training[3] data
	X_2, t_2 = flatten(data['training'][2])
	X_3, t_3 = flatten(data['training'][3])

	# fill t array with 2's and 3's
	t_2 = np.array(np.full((X_2.shape[0], 1), 2))
	t_3 = np.array(np.full((X_3.shape[0], 1), 3))
	X = np.concatenate((X_2, X_3))
	t = np.concatenate((t_2, t_3))

	# flatten testing[2] and testing[3] data
	X_2_test, t_2_test = flatten(data['testing'][2])
	X_3_test, t_3_test = flatten(data['testing'][3])

	# fill t array with 2's and 3's
	t_2_test = np.array(np.full((X_2_test.shape[0], 1), 2))
	t_3_test = np.array(np.full((X_3_test.shape[0], 1), 3))
	X_test = np.concatenate((X_2_test, X_3_test))
	t_test = np.concatenate((t_2_test, t_3_test))

	# run logistic regression to differentiate 2's and 3's
	clf = lin.LogisticRegression()
	clf.fit(X, t)
	training_err = clf.score(X, t) # = 0.984448672347
	testing_err = clf.score(X_test, t_test) # = # 0.969637610186
	# print(training_err)
	# print(testing_err)

question_4(data)

plt.show()