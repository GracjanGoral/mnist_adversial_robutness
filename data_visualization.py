adv_trained_madry_accurance_u = [93.76, 94.65, 95.47, 96.42, 97.31, 97.90]
adv_trained_madry_perturbation_epsilon_u = [0.3, 0.25, 0.20, 0.15, 0.10, 0.05]

adv_trained_madry_accurance_r = [92.49, 93.93, 95.10, 96.28, 97.27, 97.89]
adv_trained_madry_perturbation_angle = [0.3, 0.25, 0.20, 0.15, 0.10, 0.05]

adv_trained_madry_accurance_r = [4.43]
adv_extra_angle = [45]

adv_trained_madry_accurance_s = [93.90, 94.69, 95.38, 96.31, 97.21, 97.88]
adv_trained_madry_perturbation_shift = [0.3, 0.25, 0.20, 0.15, 0.10, 0.05]

adv_trained_madry_accurance_g = [94.21, 94.95, 95.59, 96.47, 97.23, 97.87]
adv_trained_madry_perturbation_gauss = [0.3, 0.25, 0.20, 0.15, 0.10, 0.05]


grando_adv_normal_ua = [0.00, 0.00, 0.00, 0.18, 11.61, 80.52]
grando_adv_normal_up= [0.3, 0.25, 0.20, 0.15, 0.10, 0.05]

grando_adv_normal_ra = [0.00, 0.00, 0.01, 0.14, 10.96, 80.27]
grando_adv_normal_rp = [0.3, 0.25, 0.20, 0.15, 0.10, 0.05]

grando_adv_normal_sa = [0.00, 0.00, 0.01, 0.18, 11.42, 80.40] 
grando_adv_normal_sp = [0.3, 0.25, 0.20, 0.15, 0.10, 0.05]

grando_adv_normal_ga = [0.00, 0.00, 0.01, 0.18, 11.83, 80.55]
grando_adv_normal_gp = [0.3, 0.25, 0.20, 0.15, 0.10, 0.05]

#ten plik pokazuje graficznie i tekstowo wyniki eksperymentów przeprowadzonych na "MNIST" z różnymi zakłóceniami;

import pandas as pd
import matplotlib.pyplot as plt

transform = ["translation: 0.3", "translation: 0.05", "rotation: 0.3", "rotation: 0.05", "gauss nois: mean = 0, varaince = 0.3", "gauss nois: mean = 0, varaince = 0.05", "uniform nois: espilon = 0.3", "uniform nois: espilon = 0.05"]
value = [93.90, 97.88, 92.49, 97.89, 94.21,97.87, 93.76, 97.90]

pars = list(zip(transform, value))   

print("M-network adversial pre-trained")
print("Accuracy on test data")
df = pd.DataFrame(data = pars, columns=['transformations', 'accuracy [%]'])
#wizualizacja tekstowa

print(df)

print('\n')
#wizualizacja graficzna
df_v = pd.DataFrame({'accurancy [%]': value}, index=transform)
ax = df_v.plot.barh(rot=0)

fig, axs = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Accurance on GrAnDo normal pre-trained network on diffrent perturbations')
axs[0, 0].plot(grando_adv_normal_up, grando_adv_normal_ua, '--o', mfc='red')
axs[0, 0].set_xlabel('size of perturbation')
axs[0, 0].set_ylabel('accuracy [%]')
axs[0, 0].set_title("uniformlay noise")

axs[0, 1].plot(grando_adv_normal_rp, grando_adv_normal_ra, '--o', mfc='red')
axs[0, 1].set_xlabel('size of perturbation')
axs[0, 1].set_ylabel('accuracy [%]')
axs[0, 1].set_title('rotation noise')

axs[1, 0].plot(grando_adv_normal_sp, grando_adv_normal_sa, '--o', mfc='red')
axs[1, 0].set_xlabel('size of perturbation')
axs[1, 0].set_ylabel('accuracy [%]')
axs[1, 0].set_title('shift noise')

axs[1, 1].plot(grando_adv_normal_gp, grando_adv_normal_ga, '--o', mfc='red')
axs[1, 1].set_xlabel('size of perturbation')
axs[1, 1].set_ylabel('accuracy [%]')
axs[1, 1].set_title('gauss noise')

plt.subplots_adjust(top=0.85, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.35)
