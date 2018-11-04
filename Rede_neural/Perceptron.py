import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split as split
import array

def modelar_dados ():
	dataset = pd.read_csv('leaf.csv', header = None)
	
	classes = dataset[dataset.columns[0]].values

	dataset.drop([0], axis = 1, inplace = True)
	'''
	individuos = np.vstack((dataset[dataset.columns[0]].values,
							dataset[dataset.columns[1]].values,
							dataset[dataset.columns[2]].values,
							dataset[dataset.columns[3]].values,
							dataset[dataset.columns[4]].values,
							dataset[dataset.columns[5]].values,
							dataset[dataset.columns[6]].values,
							dataset[dataset.columns[7]].values,
							dataset[dataset.columns[8]].values,
							dataset[dataset.columns[9]].values,
							dataset[dataset.columns[10]].values,
							dataset[dataset.columns[11]].values,
							dataset[dataset.columns[12]].values,
							dataset[dataset.columns[13]].values,))
	'''
	
	ind_t, ind_te, cl_t, cl_te = split(dataset, classes, test_size = 0.8, shuffle = False)

	return ind_t, ind_te, cl_t, cl_te

def main():
	ind_treino, ind_teste, cl_treino, cl_teste = modelar_dados()

	perc = Perceptron(verbose=0, eta0 = 0.1, max_iter=100000)

	perc.fit(ind_treino, cl_treino)

	print ('Precisão Base de treino: {0:.2f}%'.format(perc.score(ind_treino,cl_treino)*100))

if __name__ == '__main__':
	main()
