import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

multiclass = np.array([[2, 1, 0, 0],
                       [1, 2, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [2, 1, 0, 0],
                       [1, 2, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       ])

class_names = ['class a', 'class b', 'class c', 'class d',
               'class a', 'class b', 'class c', 'class d']

fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                class_names=class_names)
plt.show()
