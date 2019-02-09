import numpy as np
import scipy 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

def grando_mnist(batch_size, mnist):
    """ input: należy podać liczbę zdjęć, które chcemy uzyskać;
output: para: (macierz rozmiaru liczba zdjęć x 28 x 28, odpowiadająca jej etykieta);
nalży importować zbiór mnist: 
from tensorflow.examples.tutorials.mnist import input_data"""
    
    assert batch_size > 0
    table_of_image = []
    x_batch, y_batch = mnist.train.next_batch(batch_size) #y_batch jest etykietą danej cyfry, w tej funkcji jest nam nieporzebne 
    
    return (x_batch.reshape(batch_size, 28, 28), y_batch)



def grando_imshow(data, rows, columns):
    """input: data: (ndarray, label) np. dla mnist, (macierz 28x28, etykieta);
rows, columns: wiersze i kolmuny w których mają być umieszczone obrazy; należy pamiętać, że
rows x columns = moc_zbioru(data)
output: figure 
    """
    assert data != ()
    assert rows > 0 and columns > 0
    assert rows*columns == len(data[0])
    
    d = data
    counter = 0
    
    if len(d[0]) == 1:
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(d[0][0])
        ax.set_title("true label:" + " " + str(d[1][0]))
                     
    else:
                     
        fig, axs = plt.subplots(nrows=rows, ncols=columns, constrained_layout=True)
        for ax in axs.flat:
            ax.imshow(d[0][counter])
            ax.set_title("true label:" + " " + str(d[1][counter]))
            counter += 1
    
    return fig  

def grando_transform_rotation(image, angle):
    """ input: macierz (ndarray), kąt (float) o jaki chcemy obrócić obraz (jeszcze nie wiem, gdzie leży punkt wokół, którego obracamy figurę)
output: macierz (ndarray)"""
    return scipy.ndimage.rotate(image, angle)

def grando_transform_shift(image, shift):
    """ input: macierz (ndarray), shift (float) odpowiedzialny jest za jednakowe przesunięcie figury wzdłoż osi OX i OY;
output: macierz (ndarray)""" 
    return scipy.ndimage.shift(image, shift)

def grando_transform_zoom(image, zoom):
    """ input: macierz (ndarray), zoom (float) odpowiedzialny jest za jednakowe powiększenie figury względem osi OX i OY; patrzcie na osie;
output: macierz (ndarray)"""
    return scipy.ndimage.zoom(image, zoom)

def grando_transform_gauss(image, mean, variance):
    """ input: macierz (ndarray), mean (float), varaince (float):
output: macierz (ndarray) z dodanym zaburzeniem gaussowskim"""
    return image + np.random.normal(mean, variance, image.shape)

def grando_transform_color(image, col):
    """ input: macierz (ndarray), col (int) odpowiedzialny jest za zmianę koloru obrazu (szczegóły niech spoczywają w dokumentacji);
wątpie by to wpłynęło jakoś na skuteczność rozpoznawania, ale ładnie wygląda; zakres zmiennej "col" to 0..11 (liczby naturalne);
output: macierz (ndarray)"""
    
    t = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    
    assert col > -1 and col < len(t)
   
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(image).set_cmap(t[col])
    return fig
