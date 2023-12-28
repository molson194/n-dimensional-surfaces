import numpy as np

# Input data
x = [8,-44,17,-75,18,-92,11,1,-80,1,-40,28,-60,-71]
y = [65,-51,-33,48,45,-41,13,17,-64,-31,71,51,-20,-71]
r = [39,-84,-18,49,-29,16,73,-98,-57,-12,17,-43,1,-49]

z  = [x[i]*x[i]/10 + 5*y[i] + x[i]*y[i]/6 + r[i] for i in range(len(x))]

# Derivative matrix multiplication
sum_1 = len(x)
sum_x = sum(x)
sum_y = sum(y)
sum_x_2 = sum([x[i]*x[i] for i in range(len(x))])
sum_x_y = sum([x[i]*y[i] for i in range(len(x))])
sum_y_2 = sum([y[i]*y[i] for i in range(len(x))])
sum_x_3 = sum([x[i]*x[i]*x[i] for i in range(len(x))])
sum_x_2_y = sum([x[i]*x[i]*y[i] for i in range(len(x))])
sum_x_y_2 = sum([x[i]*y[i]*y[i] for i in range(len(x))])
sum_y_3 = sum([y[i]*y[i]*y[i] for i in range(len(x))])
sum_x_4 = sum([x[i]*x[i]*x[i]*x[i] for i in range(len(x))])
sum_x_3_y = sum([x[i]*x[i]*x[i]*y[i] for i in range(len(x))])
sum_x_2_y_2 = sum([x[i]*x[i]*y[i]*y[i] for i in range(len(x))])
sum_x_y_3 = sum([x[i]*y[i]*y[i]*y[i] for i in range(len(x))])
sum_y_4 = sum([y[i]*y[i]*y[i]*y[i] for i in range(len(x))])

sum_z = sum(z)
sum_x_z = sum([x[i]*z[i] for i in range(len(x))])
sum_y_z = sum([y[i]*z[i] for i in range(len(x))])
sum_x_2_z = sum([x[i]*x[i]*z[i] for i in range(len(x))])
sum_x_y_z = sum([x[i]*y[i]*z[i] for i in range(len(x))])
sum_y_2_z = sum([y[i]*y[i]*z[i] for i in range(len(x))])

# Solve matrix
A = np.array([
    [sum_1, sum_x, sum_y, sum_x_2, sum_x_y, sum_y_2],
    [sum_x, sum_x_2, sum_x_y, sum_x_3, sum_x_2_y, sum_x_y_2],
    [sum_y, sum_x_y, sum_y_2, sum_x_2_y, sum_x_y_2, sum_y_3],
    [sum_x_2, sum_x_3, sum_x_2_y, sum_x_4, sum_x_3_y, sum_x_2_y_2],
    [sum_x_y, sum_x_2_y, sum_x_y_2, sum_x_3_y, sum_x_2_y_2, sum_x_y_3],
    [sum_y_2, sum_x_y_2, sum_y_3, sum_x_2_y_2, sum_x_y_3, sum_y_4]])
B = np.array([
    sum_z,
    sum_x_z,
    sum_y_z,
    sum_x_2_z,
    sum_x_y_z,
    sum_y_2_z])
C = np.linalg.solve(A,B)

# Print equation
print('----------------------------------------------------------')
print('First pass')
print(f'{C[0]:.{2}f} + {C[1]:.{2}f}x+ {C[2]:.{2}f}y + {C[3]:.{2}f}x^2 + {C[4]:.{2}f}xy + {C[5]:.{2}f}y^2')
print('----------------------------------------------------------')

c0_sum = abs(C[0]) * len(x)
c1_sum = sum([abs(C[1]*x[i]) for i in range(len(x))])
c2_sum = sum([abs(C[2]*y[i]) for i in range(len(x))])
c3_sum = sum([abs(C[3]*x[i]*x[i]) for i in range(len(x))])
c4_sum = sum([abs(C[4]*x[i]*y[i]) for i in range(len(x))])
c5_sum = sum([abs(C[5]*y[i]*y[i]) for i in range(len(x))])

print('Term contribution')
print(f'{c0_sum:.{2}f}, {c1_sum:.{2}f}, {c2_sum:.{2}f}, {c3_sum:.{2}f}, {c4_sum:.{2}f}, {c5_sum:.{2}f}')
print('----------------------------------------------------------')

D = np.array([
    [sum_y_2, sum_x_2_y, sum_x_y_2],
    [sum_x_2_y, sum_x_4, sum_x_3_y],
    [sum_x_y_2, sum_x_3_y, sum_x_2_y_2]
])
E = np.array([
    sum_y_z,
    sum_x_2_z,
    sum_x_y_z])
F = np.linalg.solve(D,E)
print('Second pass')
print(f'{F[0]:.{2}f}y + {F[1]:.{2}f}x^2+ {F[2]:.{2}f}xy')
print('----------------------------------------------------------')

z_hat = [F[0]*y[i]+F[1]*x[i]*x[i]+F[2]*x[i]*y[i] for i in range(len(x))]
print ("{:<8} {:<8} {:<10} {:<10}".format('x','y','z','z_est'))
for i in range(len(z)):
    print("{:<8} {:<8} {:<10} {:<10}".format(x[i], y[i], round(z[i],2), round(z_hat[i],2)))
