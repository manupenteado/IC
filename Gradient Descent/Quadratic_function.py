import numpy as np
import matplotlib.pyplot as plt

def funcao_y(x):
    return x**2

def derivada(x):
    return 2 * x

x = np.arange(-100, 100, 0.1)
y = funcao_y(x)

posicao_atual = (80, funcao_y(80))

taxa_aprendizagem = 0.01

for _ in range (200):
   novo_x = posicao_atual[0] - taxa_aprendizagem * derivada(posicao_atual[0])
   novo_y = funcao_y(novo_x)
   posicao_atual = (novo_x, novo_y)

   plt.plot(x,y, color = "purple")
   plt.scatter(posicao_atual[0], posicao_atual[1], color = "orange")
   plt.pause(0.001)
   plt.clf()

print(f"O x correspondente ao ponto mínimo calculado é: {novo_x}.")

