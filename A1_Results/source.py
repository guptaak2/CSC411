import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin

def mymult(A,B):
	len_A = A.shape[0]
	len_B_columns = B.shape[1]
	len_B_rows = B.shape[0]
	C = np.zeros((len_A, len_B_columns))
	for i in range(len_A):
		for j in range(len_B_columns):
			for k in range(len_B_rows):
				C[i][j] += A[i][k] * B[k][j]
	return C;

def mymeasure(I,K,J):
	A = np.random.rand(I,K)
	B = np.random.rand(K,J)
	start = time.time()
	C1 = np.matmul(A,B)
	end = time.time()
	print ("matmul: %s" % (end - start))
	start_my = time.time()
	C2 = mymult(A,B)
	end_my = time.time()
	print ("my_mult: %s" % (end_my - start_my))
	print(np.sum(((C1-C2)**2)))

with open('data1.pickle', 'rb') as f:
	dataTrain, dataTest = pickle.load(f)

N_train = len(dataTrain)
N_test = len(dataTest)
X_train = dataTrain[:,0]
T_train = dataTrain[:,1]
X_test = dataTest[:,0]
T_test = dataTest[:,1]

with open('data2.pickle','rb') as f:
	dataVal, dataTestReg = pickle.load(f)

N_val = len(dataVal)
N_test_reg = len(dataTestReg)
X_val = dataVal[:,0]
T_val = dataVal[:,1]
X_test_reg = dataTestReg[:,0]
T_test_reg = dataTestReg[:,1]

def dataMatrix(X, M):
	Z = np.empty((X.shape[0], M+1))
	for n in range(0, X.shape[0]):
		for m in range(0, M+1):
			Z[n][m] = (X[n])**m
	return Z;

def fitPoly(M):
	Z = dataMatrix(X_train, M)
	output = np.linalg.lstsq(Z, T_train)
	weights = np.poly1d(np.flip(output[0],0))
	predict_train = weights(X_train)
	error_train = np.mean((T_train - predict_train)**2)
	predict_test = weights(X_test)
	error_test = np.mean((T_test - predict_test)**2)
	return output[0], error_test, error_train

def plotPoly(w):
	weights = np.poly1d(np.flip(w, 0))
	predict_train = weights(X_train)
	predict_test = weights(X_test)
	plt.figure(1)
	plt.plot(X_test, predict_test, 'r.', label = 'Fitted line')
	plt.plot(X_train, T_train, 'bo', label = 'Original data')
	plt.xlim(0, 1)
	plt.xticks(np.linspace(0, 1, 1000, endpoint = True))
	plt.ylim(-15, 15)
	return predict_test

def bestPoly():
	fig, axes = plt.subplots(nrows=4, ncols=4, sharex = True, sharey = True)
	axes = axes.flat
	for M in range(0, 16):
		weights, errors_test, errors_train = fitPoly(M)
		print("M = %s; testing error = %s; training error = %s" %(M, errors_test, errors_train))

		axes[M].plot(X_test, plotPoly(weights), 'r.', linewidth = 1)
		axes[M].plot(X_train, T_train, 'b.', linewidth = 1)
	
		plt.figure(2)
		plt.xlim(0, 15)
		plt.ylim(0, 250)
		plt.xlabel('M')
		plt.ylabel('Error')
		plt.grid('on')
		plt.title('Question 3: Training and Test error')
		plt.plot(M, errors_train, 'b.')
		plt.plot(M, errors_test, 'r.')

	plt.show()

	plt.title("Question 3: best-fitting polynomial (degree = 4)")
	plt.xlabel("x")
	plt.ylabel("t")
	plt.plot(plotPoly(fitPoly(4)[0]))
	plt.show()
	
def fitRegPoly(M, alpha):
	ridge = lin.Ridge(alpha)
	Z = dataMatrix(X_train, M)
	ridge.fit(Z, T_train)
	weights = ridge.coef_
	weights[0] = ridge.intercept_
	predict_train = ridge.predict(Z)
	error_train = np.mean((T_train - predict_train)**2)
	predict_val = ridge.predict(dataMatrix(X_val, M))
	error_val = np.mean((T_val - predict_val)**2)
	predict_test = ridge.predict(dataMatrix(X_test, M))
	error_test = np.mean((T_test - predict_test)**2) 
	return weights, error_train, error_val, predict_test, error_test


def plotRegPoly(w):
	weights = np.poly1d(np.flip(w, 0))
	predict_test = weights(X_test)
	plt.figure(1)
	plt.plot(X_test, predict_test, 'r.', label = 'Fitted line')
	plt.plot(X_train, T_train, 'b.', label = 'Original data')
	plt.xlim(0, 1)
	plt.xticks(np.linspace(0, 1, 1000, endpoint = True))
	plt.ylim(-15, 15)
	return predict_test

def bestRegPoly():
	fig2, axes2 = plt.subplots(nrows=4, ncols=4, sharex = True, sharey = True)
	axes2 = axes2.flat
	for alpha in range(-13, 3):
		weights, error_train, error_val, predict_test, error_test = fitRegPoly(15, 10**(alpha))
		print("alpha = %s; training error = %s; validation error = %s; test error = %s" %(alpha, error_train, error_val, error_test))

		axes2[alpha].plot(X_test, predict_test, 'r.', linewidth = 1)
		axes2[alpha].plot(X_train, T_train, 'b.', linewidth = 1)
	
		plt.figure(3)
		plt.ylim(0, 250)
		plt.xlabel('alpha')
		plt.ylabel('Error')
		plt.grid('on')
		plt.title('Question 4: Training and Validation error')
		plt.semilogx(10**alpha, error_train, 'b.')
		plt.semilogx(10**alpha, error_val, 'r.')

	plt.show()

	plt.title("Question 4: best-fitting polynomial (alpha = -5)")
	plt.xlabel("x")
	plt.ylabel("t")
	plt.plot(plotRegPoly(fitRegPoly(15, 10**-5)[0]))
	plt.show()


#mymeasure(1000,50,100)
#mymeasure(1000,1000,1000)
#bestPoly()
#bestRegPoly()

# def regGrad(Z, t, w, alpha):
# 	reg = np.multiply(2*alpha, w)
# 	y = w[0]
# 	i = 1
# 	while i < N_train:
# 		y += (np.multiply(X_train[i], w[i]))
# 		i += 1

# 	predict_train = np.array(y)
# 	print(predict_train)
# 	return (reg - (2 * np.sum(np.subtract(T_train, y)) * Z))

# Z = dataMatrix(X_train, 15)
# t = T_train
# w = fitRegPoly(15, 10**-3)[0]
# alpha = 10**-3

# print(w)
# answer = regGrad(Z, t, w, alpha)

	







