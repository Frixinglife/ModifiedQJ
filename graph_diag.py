import matplotlib.pyplot as plt

data_n = [int(i) for i in range(32)]

data_diag_f = []
with open("matrix_diag_float.txt") as f:
	for line in f:
		data_diag_f.append([float(x) for x in line.split()])

data_diag_d = []
with open("matrix_diag_double.txt") as f:
	for line in f:
		data_diag_d.append([float(x) for x in line.split()])

data_diag_h = []
with open("matrix_diag_half.txt") as f:
	for line in f:
		data_diag_h.append([float(x) for x in line.split()])
         
plt.title("Диагональ матрицы плотности ρ")
plt.xlabel("Номер элемента, i")
plt.ylabel("ρ(i,i)")

plt.plot(data_n, data_diag_h, label = 'Половинная точность', color = 'm')
# plt.plot(data_n, data_diag_f, label = 'Одинарная точность', color = 'b')
# plt.plot(data_n, data_diag_d, label = 'Двойная точность', color = 'g')

plt.legend()
plt.grid(True, linestyle='-', color='0.75')

plt.show()