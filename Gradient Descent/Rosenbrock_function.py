import numpy as np
import matplotlib.pyplot as plt 

#defining the function
def funcao_z(x, y):
    return ((1 - x)**2) + (100 * (y - (x**2))**2)

#calculating the partial derivatives
def calular_gradiente(x, y):
    return ((-2) * (1 - x)) - (400 * x * (y - x**2)), 200 * (y - x**2)

#defining values to build the graph
x = np.arange (-2, 2, 0.1)
y = np.arange (-1, 3, 0.1)

#creating two 2D matrices from the vectors x and y
X, Y = np.meshgrid(x, y)

#defining a variable to the function (x,y)
Z = funcao_z (X, Y)

#defining an initial point
pos_atual = (-1, -1, funcao_z(-1, -1))

#defining a learning rate
taxa_aprendizagem = 0.001

#ax é uma variavel que recebe o objeto 3d criado
#computed_zorder = False -> so we can see it, without it, the function is always the priority
#zorder defines the priority
ax = plt.subplot(projection = "3d", computed_zorder = False)

#quanto maior o número de repetições, mais perto do mínimo real vai se chegar
#the higher number of repetitions, the closer it will get to the real minimum
for _ in range (100):

    #receving the values of the derivatives corresponding to each point
    vetor_grade = calular_gradiente(pos_atual[0], pos_atual[1])
    derivada_x = vetor_grade[0]
    derivada_y = vetor_grade[1]

    #if the gradient vector (which contains the derivatives) is too small, it interrupts
    if np.linalg.norm(vetor_grade) < 1e-6:
        print("Derivada se aproximou muito de zero.")
        break

    #se o vetor gradiente for maior que 10**-6, o x e y vao assumir novos valores
    #se aproximam do ponto minimo atravez do produto entre a taxa de aprendizagem e a posicao atual
    novo_x, novo_y = pos_atual[0] - taxa_aprendizagem * derivada_x, pos_atual[1] - taxa_aprendizagem * derivada_y
    pos_atual = (novo_x, novo_y, funcao_z(novo_x, novo_y))

    #criando o gráfico
    
    ax.plot_surface(X, Y, Z, cmap = "plasma", zorder = 0)
    ax.scatter(pos_atual[0], pos_atual[1], pos_atual[2], color = "blue", zorder = 1)
    plt.pause(0.001)
    ax.clear()
    

    #fim do for

#pegando os ultimos valores de x e y (os valores minimos calculados) e arredondando para 4 casas decimais
x_minimo = round (novo_x, 4)
y_minimo = round (novo_y, 4)

print(f"O mínimo global da função é aproximadamente em = ({x_minimo}, {y_minimo})")

