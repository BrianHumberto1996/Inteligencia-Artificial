entradas = ((0, 0), (0 ,1), (1, 0), (1, 1))

pesos = (-10, 20, 20)

print('g({0} + ({1} * x1) + ({2} * x2)'.format(pesos[0], pesos[1], pesos[2]))

for i in entradas:
    res = pesos[0] + (20 * i[0] + (20 * i[1]))
  
print('g({0} + ({1} * {2}) + ({3} * {4})) = {5}'.format(pesos[0], pesos[1], i[0], pesos[2], i[1], res))