### IMPORTS ################################################

import pandas as pd
import numpy as np

### CONSTANTES #############################################

NUM_EPOCAS = 1000
BIAS = 1
TAXA_APRENDIZADO = 0.1
ATRIBUTO = 14

### MODELAR DADOS ##########################################

def modelar_dados():
	### FUNCAO PARA IMPORTAR OS DADOS E GERAR OS VETORES DE INDIVIDUOS E AS RESPECTIVAS CLASSES ###

	dataset_treino = pd.read_csv('leaf.csv')

	classes_treino = dataset_treino['Classe'].values
	
	dataset_treino.drop(['Specimen'], axis=1, inplace=True)
	dataset_treino.drop(['Classe'], axis=1, inplace=True)

	(dataset_treino.columns[2])

	individuos_treino = np.vstack((  dataset_treino[dataset_treino.columns[0]].values,
									dataset_treino[dataset_treino.columns[1]].values,
									dataset_treino[dataset_treino.columns[2]].values,
									dataset_treino[dataset_treino.columns[3]].values,
									dataset_treino[dataset_treino.columns[4]].values,
									dataset_treino[dataset_treino.columns[5]].values,
									dataset_treino[dataset_treino.columns[6]].values,
									dataset_treino[dataset_treino.columns[7]].values,
									dataset_treino[dataset_treino.columns[8]].values,
									dataset_treino[dataset_treino.columns[9]].values,
									dataset_treino[dataset_treino.columns[10]].values,
									dataset_treino[dataset_treino.columns[11]].values,
									dataset_treino[dataset_treino.columns[12]].values,
									dataset_treino[dataset_treino.columns[13]].values))

	return individuos_treino, classes_treino

### FUNCAO DE AVALIACAO ####################################

def func_ativacao(valor):
	### FUNCAO DE AVALIACAO DO PERCEPTRON: ESTÁ CONFIGURADO PARA 2 SAIDAS, FALTA MODELAR PARA MAIS POSSIBILIDADES ###

	if valor < 0:
		return -1
	else:
		return  1

### MAIN ###################################################

def main():
	### FUNCAO PRINCIPAL, REALIZA O CALCULO DOS ERROS E CORRIGE OS PESOS DE ACORDO COM O ERRO CALCULADO ###

	ind_treino, classes_treino = modelar_dados()

	pesos = np.zeros([1, 15])
	erro  = np.zeros(14)

	for i in range (NUM_EPOCAS):
		for k in range (ATRIBUTO):

			treino_b = np.hstack((BIAS, ind_treino[:,k]))

			campo_induzido = np.dot(pesos, treino_b)

			saida_Perceptron = func_ativacao(campo_induzido)

			erro[k] = classes_treino[k] - saida_Perceptron

			pesos = pesos + TAXA_APRENDIZADO*erro[k]*treino_b

	erro = np.array(erro)
	print ('Erro:    ' + str(erro))
	print ('Classes: ' + str(classes_treino))

### INICIAR PROGRAMA #######################################

if __name__ == '__main__':
	main()
