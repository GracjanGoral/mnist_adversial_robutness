#Możecie ten "skrypt" przekopiować do colaba i powinien działać. Jest to tylko wizualizacja. 
#Wcześniej jednak należy przekopiować wszystkie funkcje, które znajdują się w pliku: "grando_function.py"
#Miłej zabawy"

x = 10 #kąt, przesunięcie, powiększenie, col
mean = 0
variance = 1
#są pewne ograniczenia na x, napisłaem je w dokumentacji każdej funkcji 
counter = 0
batch_size = 10 # liczba zdjęć 
rows = 2 # rows*col = liczba zdjęć
col = 5
k = 0 #numer zdjęcia, jest on wykorzystywany tylko w funkcji "grando_transform_color"
number_of_function = 3

transform_functions = [grando_transform_rotation, grando_transform_shift, grando_transform_zoom, grando_transform_gauss]

pictures = grando_mnist(batch_size, mnist)

show = grando_imshow(pictures, rows, col)
show.suptitle("orginal photos", fontsize=16)

fig, axs = plt.subplots(nrows=rows, ncols=col, constrained_layout=True)

for ax in axs.flat:
    if number_of_function == 3:
        ax.imshow(np.clip(transform_functions[number_of_function](pictures[0][counter], mean, variance), 0, 1))
        ax.set_title("true label:" + " " + str(pictures[1][counter]))
        counter += 1
    else:
        ax.imshow(transform_functions[number_of_function](pictures[0][counter], x))
        ax.set_title("true label:" + " " + str(pictures[1][counter]))
        counter += 1
fig.suptitle("photos after transformation", fontsize=16)
plt.show()

# image_color = grando_transform_color(pictures[0][k], x)
