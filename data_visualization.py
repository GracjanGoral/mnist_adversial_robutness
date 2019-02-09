#ten plik pokoazuje graficznie i tekstowo wyniki eksperymentów przeprowadzonych na "MNIST" z różnymi zakłóceniami;

import pandas as pd
import matplotlib.pyplot as plt

transform = ["translation: 1.0", "translation: 5.0", "rotation: 10", "rotation: 45", "gauss nois: mean = 0, varaince = 1", "gauss nois: mean = 0.5, varaince = 1", "uniform nois: espilon = 0.3", "uniform nois: espilon = 0.1"]
value = [16.04, 0.01, 19.87, 4.43, 98.40, 0.01, 93.85, 97.28]

pars = list(zip(transform, value))   

df = pd.DataFrame(data = pars, columns=['transformations', 'accuracy [%]'])
#wizualizacja tekstowa
print(df)

#wizualizacja graficzna
df_v = pd.DataFrame({'accurancy [%]': value}, index=transform)
ax = df_v.plot.barh(rot=0)
