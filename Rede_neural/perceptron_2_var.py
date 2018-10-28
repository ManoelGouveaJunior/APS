#coding:utf-8
#perceptron_2_var.py

import numpy as np
import matplotlib as mat

'''
Programa de teste para treino de perceptron proposto em sala na disciplina de SII, a lógica está em realizar o treino para determinar os valores de peso adequados para uma lógica com duas entradas e a saida é dada por (X1 and (not: ¬)X2).

A saber os valores ideias de pesos encontrados manualmente foram:
	w0 (bias) = -0,6
	w1	  =  1,4
	w2	  = -1,3

Usado como dataset o arquivo percetron_2_var.csv
'''
ENTRADAS = 2

def ler_arquivo (path):

	arq = open (path, 'r')
	linhas = arq.readlines()
	header = []
	individuos = []
	classe = []
	
	for linha in linhas:

		if (linha == linhas[0]):
			header = linha
		else:
			individuos.append(linha[:-3])
			classe.append(linha[-2:-1])

	arq.close()

	return header,individuos,classe

	#FIM ler_arquivo

def verifica_erro(pesos,individuo,classe):
	soma = 0
	func = 0

	x1 = int(individuo[:1])
	x2 = int(individuo[2:])
		
	soma += x1*pesos[1] + x2*pesos[2] + pesos[0]

	#Função avaliativa
	if (soma <= 0):
		func = 0
	else:
		func = 1

	return int(classe) - func

	#FIM verifica_erro

def corrige_pesos(pesos, taxa, erro, individuo):

	x = [1, int(individuo[:1]), int(individuo[2:])]

	peso_aux = pesos[:]

	for i in range (len(pesos)):
		peso_aux[i] = pesos[i] + (taxa*erro*x[i])
		
	return peso_aux

	#FIM corrige_pesos
	

def main():

	pesos = [0.2, 1.4, -0.5]
	taxa = 0.8
	treinando = True
	
	header, individuos, classes = ler_arquivo('perceptron_2_var.csv')
	arq = open('pesos_perceptron_2_var.txt', 'w')

	arq.write(str(pesos) + '\n')

	while (treinando == True):
		erros = []

		for cont in range (0, len(individuos)):
			erro = verifica_erro(pesos,individuos[cont],classes[cont])

			if erro == 1 or erro == -1:
				pesos = corrige_pesos(pesos, taxa, erro, individuos[cont])
				arq.write(str(pesos) + '\n')

		for j in range (0, len(individuos)):
			erros.append(verifica_erro(pesos,individuos[j],classes[j]))

		if -1 in erros or 1 in erros:
			treinando = True
		else:
			treinando = False


	arq.close()

if __name__ == "__main__":
	main()






















