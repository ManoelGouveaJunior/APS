import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.model_selection import train_test_split as split
import time

### CONSTANTES ###
ALGORITMO = 'NAIVE_BAYES'
TIPO_NB = 'MULTINOMIAL'
TAM_TESTES = 0.9
IND = 40

def modelar_dados (teste):
        ### FUNCAO PARA MODELAR OS DADOS E DIVIDI-LOS EM UMA BASE DE TREINO E DE TESTES DE ACORDO COM O PRAMETRO TAM_TESTES INFORMADO NAS CONSTANTES ###

	dataset = pd.read_csv('leaf.csv', header = None)
	
	classes = dataset[dataset.columns[0]].values

	dataset.drop([0], axis = 1, inplace = True)
	
	ind_t, ind_te, cl_t, cl_te = split(dataset, classes, test_size = teste, shuffle = True)

	return ind_t, ind_te, cl_t, cl_te

        ### FIM MODELAR_DADOS ###

def main():
    ### FUNCAO PRINCIPAL ###

    ini_tot = time.time()
    
    arq = open('RELAT_DESEMPENHO_{0}_{1}_{2}_.txt'.format(ALGORITMO, TIPO_NB, TAM_TESTES), 'w')

    parametros=['################# PARAMETROS #################\n',
                'Algoritmo de Cllassificacao: {0}_{1}\n'.format(ALGORITMO,TIPO_NB),
                'Numero de individuos.......: {0}\n'.format(str(IND)),
                'Base de Treino.............: {0:.2f}%\n'.format((1 - TAM_TESTES)*100),
                'Base de Testes.............: {0:.2f}%\n'.format(TAM_TESTES*100),
                '################# PARAMETROS #################\n']

    arq.writelines(parametros)

    for c in range(50):
        inicio = time.time()
        
        errou = 0
        acertou = 0
        
        dados = []
        
        ind_treino, ind_teste, cl_treino, cl_teste = modelar_dados(TAM_TESTES)

        naive = nb()

        naive.fit(ind_treino, cl_treino)

        x = naive.predict (ind_teste)

        for i in range (len(x)):
                            if cl_teste[i] == x[i]:
                                    acertou = acertou + 1
                            else:
                                    errou = errou + 1

        fim = time.time()

        tempo = fim - inicio

        dados.append('\n############## RESULTADO TESTE {0} #############\n'.format(str(c+1)))
        dados.append('Vetor algoritmo: {0} // Tamanho do vetor: {1}\n'.format(str(x),str(len(x))))
        dados.append('Vetor gabarito.: {0} // Tamanho do vetor: {1}\n'.format(str(cl_teste),str(len(cl_teste))))
        dados.append('Precisao do treinamento: {0:.2f}%\n'.format(naive.score(ind_treino,cl_treino)*100))
        dados.append('Acertos................: {0:.2f}% ({1} acertos)\n'.format(acertou*100/len(x), acertou))
        dados.append('Erros..................: {0:.2f}% ({1} erros)\n'.format(errou*100/len(x), errou))
        dados.append('Tempo de execução......: {0:.2f}s\n'.format(tempo))
        dados.append('################# FIM TESTE {0} ################\n'.format(str(c+1)))

        arq.writelines(dados)

    print ('FIM EXECUCAO NAIVE BAYES MULTINOMIAL')

    fim_tot = time.time()

    tempo_tot = fim_tot - ini_tot

    arq.write('\nTEMPO TOTAL DE EXECUÇÃO: {0}'.format(str(tempo_tot)))

    arq.close()

    ### FIM MAIN ###

if __name__ == '__main__':
        ### INICIA O PROGRAMA ###
	main()
