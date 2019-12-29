import tensorflow as tf
import keras 
import sklearn
import numpy 
import kmapper
import pandas
from matplotlib import pyplot
from ripser import ripser
	

from sklearn.neighbors import NearestNeighbors
from persim import plot_diagrams
from sklearn.decomposition import PCA
	
neigh = NearestNeighbors(n_neighbors=5, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)
model_djia = tf.keras.models.load_model('weights/djia.h5')
model_djia.summary()
for layer in model_djia.layers:
    if 'conv' not in layer.name:
        continue 
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)
    #print(filters.shape)
	

#print(filters)
print(model_djia.layers[5].get_weights()[0].shape)
#print(model_djia.layers[5].get_weights()[0][4][4][63])
raw_data = []
for i in range(5):
    for j in range(5):
        for k in range(64):
            raw_data.append(model_djia.layers[14].get_weights()[0][i][j][k])
raw_data_arr = numpy.array(raw_data)
print(raw_data)
#print(len(model_djia.layers[5].get_weights()[0].flatten()))
mapper = kmapper.KeplerMapper(verbose=1) #select value for verbose
neigh.fit(raw_data_arr)
#from sklearn.decomposition import PCA
pca = PCA(n_components=2) # select value for n_components
	

#data_trans = mapper.fit_transform(filters, projection=[0,1])
projected_data = mapper.project(raw_data_arr, projection=pca) #choose which projection to use
	

#projected_data = mapper.project(raw_data_arr, "knn_distance_5") #change which kind of projection to use
#lens should be equal to the data_trans or projected_data
#choose which clusterer to use
#choose which cover to use
simplicial_complex = mapper.map(projected_data)
mapper.visualize(simplicial_complex, color_function=None, custom_tooltips=None, custom_meta=None, path_html='mapper_visualization_output.html', title='Kepler Mapper', save_file=True, X=None, X_names=[], lens=None, lens_names=[], show_tooltips=True, nbins=10)
import IPython
url = 'mapper_visualization_output.html'
iframe = '<iframe src=' + url + ' width=700 height=600></iframe>'
IPython.display.HTML(iframe)
	

#mean center the filters 
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min) #change this
#visualize the filters 
n_filters, ix = 6, 1
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(3):
        ax = pyplot.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap = 'gray')
        ix += 1
pyplot.show()
pca.fit(raw_data_arr)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
#datas = numpy.random.random((100,2))
#data = simplicial_complex 
#print(data)
diagrams = ripser(projected_data)['dgms']
plot_diagrams(diagrams, show=True)

