import matplotlib.pyplot as plt
import numpy as np
import random
import os

energy = 0
dt = 0.001
t = 0
kinetic = 0
potent = 0
energy_list = []
time_list = []
k = 10
m = 10
velocity_multiply = 1
sigma = 1
temprature_list = []
velocity_list_sqr = []
potent_list = []
v_max = 5


def is_moleculs_nearby(x, y, z, l):
    for i in range(len(x)):
        for j in range(len(x)):
            r = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)
            if r <= 1 and r != 0:
                x[i] = random.uniform(0.01, l - 0.01)
                y[i] = random.uniform(0.01, l - 0.01)
                z[i] = random.uniform(0.01, l - 0.01)
                break


def find_average_value(x):
    sum_x = 0
    for i in range(len(x)):
        sum_x += x[i]
    return sum_x / len(x)


def create_display_list(x, y, z):
    display_list = []
    display_list.append([x + l, y, z])
    display_list.append([x + l, y, z + l])
    display_list.append([x + l, y + l, z])
    display_list.append([x + l, y + l, z + l])
    display_list.append([x + l, y - l, z + l])
    display_list.append([x + l, y - l, z])
    display_list.append([x + l, y - l, z - l])
    display_list.append([x + l, y, z - l])
    display_list.append([x + l, y + l, z - l])

    display_list.append([x - l, y, z])
    display_list.append([x - l, y, z + l])
    display_list.append([x - l, y + l, z])
    display_list.append([x - l, y + l, z + l])
    display_list.append([x - l, y - l, z + l])
    display_list.append([x - l, y - l, z])
    display_list.append([x - l, y - l, z - l])
    display_list.append([x - l, y, z - l])
    display_list.append([x - l, y + l, z - l])

    display_list.append([x, y, z])
    display_list.append([x, y, z + l])
    display_list.append([x, y + l, z])
    display_list.append([x, y + l, z + l])
    display_list.append([x, y - l, z + l])
    display_list.append([x, y - l, z])
    display_list.append([x, y - l, z - l])
    display_list.append([x, y, z - l])
    display_list.append([x, y + l, z - l])

    return display_list


'''def find_nearest_display_distance(disp_list, x, y, z):
    min_distance = np.sqrt((x - disp_list[0][0]) ** 2 + (y - disp_list[0][1]) ** 2 + (z - disp_list[0][2]) ** 2)
    for i in range(len(disp_list)):
        dist = np.sqrt((x - disp_list[i][0]) ** 2 + (y - disp_list[i][1]) ** 2 + (z - disp_list[i][2]) ** 2)
        if dist <= min_distance:
            min_distance = dist
    return min_distance'''


def find_cor_nearest_display(disp_list, x, y, z):
    min_distance = np.sqrt((x - disp_list[0][0]) ** 2 + (y - disp_list[0][1]) ** 2 + (z - disp_list[0][2]) ** 2)
    cor_nearest_disp = [disp_list[0][0], disp_list[0][1], disp_list[0][2]]
    for i in range(len(disp_list)):
        dist = np.sqrt((x - disp_list[i][0]) ** 2 + (y - disp_list[i][1]) ** 2 + (z - disp_list[i][2]) ** 2)
        if dist <= min_distance:
            min_distance = dist
            cor_nearest_disp = [disp_list[i][0], disp_list[i][1], disp_list[i][2], min_distance]

    return cor_nearest_disp


def moleculs_not_in_cube(x1, x2, y1, y2, z1, z2):
    if abs(x1 - x2) > l / 2 or abs(y1 - y2) > l / 2 or abs(z1 - z2) > l / 2:
        return True
    else:
        return False


def spawn_moleculs(x, y, z, n):
    for i in range(n):
        cor_x = random.uniform(sigma/4, l - sigma/4)
        cor_y = random.uniform(sigma/4, l - sigma/4)
        cor_z = random.uniform(sigma/4, l - sigma/4)
        can_add = True
        j = 0
        while j < len(x):
            r = np.sqrt((cor_x - x[j]) ** 2 + (cor_y - y[j]) ** 2 + (cor_z - z[j]) ** 2)
            if r <= 1 * sigma :
                cor_x = random.uniform(sigma/4, l - sigma/4)
                cor_y = random.uniform(sigma/4, l - sigma/4)
                cor_z = random.uniform(sigma/4, l - sigma/4)
                j = -1
            j += 1
        x.append(cor_x)
        y.append(cor_y)
        z.append(cor_z)


def get_the_velocity_moleculs(vx, vy, vz, n):
    for i in range(n):
        turn = random.choice([-1, 1])
        vx.append(np.sqrt(temprature)*turn)
        vy.append(np.sqrt(temprature)*turn)
        vz.append(np.sqrt(temprature)*turn)


def new_acceleration(x1, y1, z1, x2, y2, z2):
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    fx = []


def find_force_on_molecul(index):
    f_x = 0
    f_y = 0
    f_z = 0
    for j in range(num):
        r = np.sqrt((x_list[index] - x_list[j]) ** 2 + (y_list[index] - y_list[j]) ** 2 + (z_list[index] - z_list[j]) ** 2)
        if r != 0 and not moleculs_not_in_cube(x_list[index], x_list[j], y_list[index], y_list[j], z_list[index], z_list[j]):
            f_x += 24 * ((2 * (sigma / r) ** 14) - (1 * (sigma / r) ** 8)) * (x_list[index] - x_list[j]) * k
            f_y += 24 * ((2 * (sigma / r) ** 14) - (1 * (sigma / r) ** 8)) * (y_list[index] - y_list[j]) * k
            f_z += 24 * ((2 * (sigma / r) ** 14) - (1 * (sigma / r) ** 8)) * (z_list[index] - z_list[j]) * k

        elif r != 0 and moleculs_not_in_cube(x_list[index], x_list[j], y_list[index], y_list[j], z_list[index], z_list[j]):

            display_cor_list = create_display_list(x_list[j], y_list[j], z_list[j])
            disp_r = find_cor_nearest_display(display_cor_list, x_list[index], y_list[index], z_list[index])[3]
            cor_nearest_display = find_cor_nearest_display(display_cor_list, x_list[index], y_list[index], z_list[index])
            if disp_r != 0:
                f_x += 24 * ((2 * (sigma / disp_r) ** 14) - (1 * (sigma / disp_r) ** 8)) * (
                        x_list[index] - cor_nearest_display[0]) * k
                f_y += 24 * ((2 * (sigma / disp_r) ** 14) - (1 * (sigma / disp_r) ** 8)) * (
                        y_list[index] - cor_nearest_display[1]) * k
                f_z += 24 * ((2 * (sigma / disp_r) ** 14) - (1 * (sigma / disp_r) ** 8)) * (
                        z_list[index] - cor_nearest_display[2]) * k
    return [f_x, f_y, f_z]


def find_energy_of_system():
    kinetic = 0
    potent = 0
    for i in range(num):

        kinetic += (velocity_x_list[i] ** 2 + velocity_y_list[i] ** 2 + velocity_z_list[i] ** 2) / 2
        # here i am counting potential energy of system and then i am going to calculate the force on each particle
        for j in range(i + 1, num):
            r = np.sqrt((x_list[i] - x_list[j]) ** 2 + (y_list[i] - y_list[j]) ** 2 + (z_list[i] - z_list[j]) ** 2)

            if r != 0 and not moleculs_not_in_cube(x_list[i], x_list[j], y_list[i], y_list[j], z_list[i], z_list[j]):

                potent += 4 * ((sigma / r) ** 12 - (sigma / r) ** 6) * m

            elif r != 0 and moleculs_not_in_cube(x_list[i], x_list[j], y_list[i], y_list[j], z_list[i], z_list[j]):

                display_cor_list = create_display_list(x_list[j], y_list[j], z_list[j])
                disp_r = find_cor_nearest_display(display_cor_list, x_list[i], y_list[i], z_list[i])[3]

                potent += 4 * ((sigma / disp_r) ** 12 - (sigma / disp_r) ** 6) * m
    return kinetic + potent


def print_xyz_cor(moment):
    file_to_write = open(
        'C:/Users/Home/PycharmProjects/molecular_dynamic/500 moleculs 20.10/xyz cor/moment{}.xyz'.format(moment),
        'w')
    print('{}'.format(num), file=file_to_write)
    print('Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=pos:R:3:velo:R:3 Time={}'.format(moment),
          file=file_to_write)

    for i in range(number):
        print('{}'.format(x_list[i]), '{}'.format(y_list[i]), '{}'.format(z_list[i]), '{}'.format(velocity_x_list[i]),
              '{}'.format(velocity_y_list[i]), '{}'.format(velocity_z_list[i]), file=file_to_write)


def find_temprature(vx, vy, vz):
    v_sqr = 0
    for i in range(num):
        v_sqr += vx[i]**2 + vy[i]**2 + vz[i]**2
    temp = v_sqr / (3 * num)
    return temp


velocity_x_list = []
velocity_y_list = []
velocity_z_list = []
x_list = []
y_list = []
z_list = []


density = float(input())
temprature = float(input())

l = 10
num = int(l ** 3 * density)
cor_par = 0.01
range_cor = 0.5


spawn_moleculs(x_list, y_list, z_list, num)
get_the_velocity_moleculs(velocity_x_list, velocity_y_list, velocity_z_list, num)

number = len(x_list)
print(number)
print(len(y_list))
print(len(z_list))
print(x_list)


display_x_list = []
display_y_list = []
display_z_list = []

cor_list_full = []
for i in range(number):
    cor_list_full.append([])
velocity_x_list_in_time = []
moment = 1
file_to_write_energy = open('C:/Users/Home/PycharmProjects/molecular_dynamic/500 moleculs 20.10/energy.txt', 'w')
file_to_write_temprature = open('C:/Users/Home/PycharmProjects/molecular_dynamic/500 moleculs 20.10/temprature.txt', 'w')
while t <= 1:

    force_list = []
    for i in range(num):
        force_list.append(find_force_on_molecul(i))

    # now i get list with force on each particle and want to calculate coordinate on the next moment of time
    force_list_cur = force_list
    for p in range(num):
        x_list[p] += velocity_x_list[p] * dt + force_list_cur[p][0] * dt**2 / 2
        y_list[p] += velocity_y_list[p] * dt + force_list_cur[p][1] * dt**2 / 2
        z_list[p] += velocity_z_list[p] * dt + force_list_cur[p][2] * dt**2 / 2
        if x_list[p] > l:
            x_list[p] = 0
        if x_list[p] < 0:
            x_list[p] = l
        if y_list[p] > l:
            y_list[p] = 0
        if y_list[p] < 0:
            y_list[p] = l
        if z_list[p] > l:
            z_list[p] = 0
        if z_list[p] < 0:
            z_list[p] = l

    force_list = []
    for i in range(num):
        force_list.append(find_force_on_molecul(i))

    for p in range(num):
        velocity_x_list[p] += (force_list_cur[p][0] + force_list[p][0]) * dt / 2
        velocity_y_list[p] += (force_list_cur[p][1] + force_list[p][1]) * dt / 2
        velocity_z_list[p] += (force_list_cur[p][2] + force_list[p][2]) * dt / 2

    for i in range(number):
        pair = [x_list[i], y_list[i], z_list[i]]
        cor_list_full[i].append(pair)

    temp = find_temprature(velocity_x_list, velocity_y_list, velocity_z_list)
    energy = find_energy_of_system()
    print_xyz_cor(moment)
    energy_list.append(energy)
    temprature_list.append(temp)

    t += dt
    moment += 1
    print(t)
    print('energy = ', energy)
    time_list.append(t)


print('energy list = ', energy_list)

for i in range(len(energy_list)):
    print(energy_list[i], file=file_to_write_energy)
    print(temprature_list[i], file=file_to_write_temprature)

molecul_list_velocity_module = []
for i in range(number):
    velocity_module = np.sqrt(velocity_x_list[i] ** 2 + velocity_y_list[i] ** 2 + velocity_z_list[i] ** 2)
    # if velocity_module <= 1:
    molecul_list_velocity_module.append(velocity_module * velocity_multiply)

molecul_list_velocity_module.sort()

average_value_of_velocity = find_average_value(molecul_list_velocity_module)

delta = (molecul_list_velocity_module[-2] - molecul_list_velocity_module[1]) / np.sqrt(number)

print('delta = ', delta)
print('molecul_velocity_list = ', molecul_list_velocity_module)
print('average value = ', average_value_of_velocity)

number = len(molecul_list_velocity_module)
print('number = ', number)

velocity_coord = molecul_list_velocity_module[0]
velocity_cor_list = []
amount_in_delta_list = np.histogram(molecul_list_velocity_module, bins=10, density=True)

for i in range(10):
    velocity_coord += delta
    velocity_cor_list.append(velocity_coord)
print('velocity_cor_list = ', velocity_cor_list)
print('amount_in_delta_list = ', amount_in_delta_list)

fig1 = plt.subplot(711)
fig1.plot(time_list, energy_list)
fig1.set_xlabel('time', fontsize=10)
fig1.set_ylabel('energy', fontsize=10)

fig2 = plt.subplot(713)
fig2.hist(molecul_list_velocity_module, bins=150, density=True)
fig2.set_ylabel('density of velocity', fontsize=10)

# fig3 = plt.subplot(513)


# fig3.plot(time_list, velocity_x_center_mass_list, color='green')
# fig3.set_xlabel('time', fontsize=10)
# fig3.set_ylabel('velocity x of center mass', fontsize=10)


fig4 = plt.subplot(715)
fig4.plot(time_list, temprature_list)
fig4.set_xlabel('time', fontsize=10)
fig4.set_ylabel('temprature', fontsize=10)


plt.show()

from tkinter import *
import random

master = Tk()
i = 0
canvas1 = Canvas(master, width=400, height=400, bg='white')
canvas2 = Canvas(master, width=400, height=400, bg='yellow')
canvas3 = Canvas(master, width=400, height=400, bg='pink')

particles1 = []
particles2 = []
particles3 = []

print(len(cor_list_full[0]))
for k in range(len(cor_list_full)):
    u = random.randint(1048576, 16777215)

    particles1.append(
        canvas1.create_oval(cor_list_full[k][0][0] * 40, cor_list_full[k][0][1] * 40, cor_list_full[k][0][0] * 40 + 5,
                            cor_list_full[k][0][1] * 40 + 5, fill='#' + str(hex(u)[2:])))
    particles2.append(
        canvas2.create_oval(cor_list_full[k][0][0] * 40, cor_list_full[k][0][2] * 40, cor_list_full[k][0][0] * 40 + 5,
                            cor_list_full[k][0][2] * 40 + 5, fill='#' + str(hex(u)[2:])))
    particles3.append(
        canvas3.create_oval(cor_list_full[k][0][1] * 40, cor_list_full[k][0][2] * 40, cor_list_full[k][0][1] * 40 + 5,
                            cor_list_full[k][0][2] * 40 + 5, fill='#' + str(hex(u)[2:])))


def game(event):
    global i
    for j in range(len(cor_list_full)):
        canvas1.coords(particles1[j], cor_list_full[j][i][0] * 40, cor_list_full[j][i][1] * 40,
                       cor_list_full[j][i][0] * 40 + 5, cor_list_full[j][i][1] * 40 + 5)
        canvas2.coords(particles1[j], cor_list_full[j][i][0] * 40, cor_list_full[j][i][2] * 40,
                       cor_list_full[j][i][0] * 40 + 5, cor_list_full[j][i][2] * 40 + 5)
        canvas3.coords(particles1[j], cor_list_full[j][i][1] * 40, cor_list_full[j][i][2] * 40,
                       cor_list_full[j][i][1] * 40 + 5, cor_list_full[j][i][2] * 40 + 5)

    if event.char == 'a':
        i += 1
    if event.char == 'd':
        i -= 1


def game1(event):
    global i
    for j in range(len(cor_list_full)):
        canvas1.coords(particles1[j], cor_list_full[j][i][0] * 40, cor_list_full[j][i][1] * 40,
                       cor_list_full[j][i][0] * 40 + 5,
                       cor_list_full[j][i][1] * 40 + 5)
        canvas2.coords(particles1[j], cor_list_full[j][i][0] * 40, cor_list_full[j][i][2] * 40,
                       cor_list_full[j][i][0] * 40 + 5,
                       cor_list_full[j][i][2] * 40 + 5)
        canvas3.coords(particles1[j], cor_list_full[j][i][1] * 40, cor_list_full[j][i][2] * 40,
                       cor_list_full[j][i][1] * 40 + 5,
                       cor_list_full[j][i][2] * 40 + 5)

    if i < len(cor_list_full[0]) - 1:
        i += 1


canvas1.grid(row=0, column=0)
canvas2.grid(row=0, column=1)
canvas3.grid(row=1, column=0)

master.bind('<KeyPress>', game)
master.bind('<Motion>', game1)
master.mainloop()
