#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
The Iris Dataset
=============================================== =======
Este conjunto de datos consta de 3 tipos diferentes de iris
(Setosa, Versicolour y Virginica) pétalo y sépalo
de longitud, almacenado en un nuzzy.

Las filas son las muestras y las columnas que son:
Longitud del sepal, ancho del sepal, longitud del pétalo y ancho del pétalo.

La gráfica de abajo usa las dos primeras características.
Consulte `aquí <https://en.wikipedia.org/wiki/Iris_flower_data_set>` _ para obtener más información.
información sobre este conjunto de datos.
"""
print(__doc__)


# Fuente de codigo: Gaël Varoquaux
#  Modificado para la documentación por Jaques Grobler
# Licencia: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# Importar algunos datos para jugar 
iris = datasets.load_iris()
X = iris.data[:, :2]  # Nosotros solo tomamos dos caracteristicas
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Trazar los puntos de entrenamiento
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Para obtener una mejor comprensión de la interacción de las dimensiones
# Traza las primeras tres dimenciones de PCA
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
