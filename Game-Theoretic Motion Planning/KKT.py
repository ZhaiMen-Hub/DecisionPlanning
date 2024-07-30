# this secipt implement a toy example of solving a standard QP with KKT condition

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 定义变量
x = ca.MX.sym('x', 2)

# 定义QP的参数
Q = ca.DM([[2, 0], [0, 2]])
c = ca.DM([-2, -5])

# 定义目标函数
objective = 0.5 * ca.mtimes([x.T, Q, x]) + ca.mtimes([c.T, x])

# 定义约束
A = ca.DM([[1, 1]])
b = ca.DM([1])
G = ca.DM([[-1, 0], [0, -1]])
h = ca.DM([0, 0])

# 定义约束
equality_constraints = A @ x - b
inequality_constraints = G @ x - h

# 使用nlpsol求解器求解QP问题
qp = {'x': x, 'f': objective, 'g': ca.vertcat(equality_constraints, inequality_constraints)}
solver = ca.nlpsol('solver', 'ipopt', qp)

# 初始猜测
x0 = ca.DM([0.5, 0.5])

# 求解
sol = solver(x0=x0, lbg=ca.DM([0, -ca.inf, -ca.inf]), ubg=ca.DM([0, 0, 0]))

# 提取结果
x_opt_qp = sol['x']

# 打印结果
print("QP解:", x_opt_qp)

# 可视化部分
x1_vals = np.linspace(-1, 2, 100)
x2_vals = np.linspace(-1, 2, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = 0.5 * (Q[0, 0] * X1**2 + 2 * Q[0, 1] * X1 * X2 + Q[1, 1] * X2**2) + c[0] * X1 + c[1] * X2


# 定义拉格朗日乘子
lambda_ = ca.MX.sym('lambda', 1)
nu = ca.MX.sym('nu', 2)

# 拉格朗日函数
Lagrangian = objective + ca.mtimes([lambda_.T, equality_constraints]) + ca.mtimes([nu.T, inequality_constraints])

# KKT条件 (equality)
grad_L_x = ca.gradient(Lagrangian, x)
complementary_slackness = ca.diag(nu) @ inequality_constraints

# 将KKT条件组合成一个系统
kkt_system = ca.vertcat(
    # equality
    grad_L_x,
    complementary_slackness,
    equality_constraints,
    # inequality
    inequality_constraints,
)
print(f'kkt_dim: {kkt_system.size1()}')
# 0 for equality and -inf for inequality
lb = ca.DM((x.size1() * 2 + equality_constraints.size1()) * [0] + inequality_constraints.size1() * [-ca.inf])

# 定义未知变量
vars = ca.vertcat(x, lambda_, nu)

# 定义NLP问题
nlp = {'x': vars, 'f': 0, 'g': kkt_system}

# 求解器选项
opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# 初始猜测
x0 = ca.DM([0.5, 0.5])
lambda0 = ca.DM([1])
nu0 = ca.DM([1, 1])
initial_guess = ca.vertcat(x0, lambda0, nu0)
lbx = ca.vertcat((x0.size1() + lambda0.size1()) * [-ca.inf], nu0.size1() * [0])

# 求解
sol = solver(x0=initial_guess, lbg=lb, ubg=0, lbx = lbx)

# 提取结果
x_opt_kkt = sol['x'][0:2]
lambda_opt_kkt = sol['x'][2]
nu_opt_kkt = sol['x'][3:]

# 打印结果
print("KKT解:", x_opt_kkt)
print("lambda:", lambda_opt_kkt)
print("nu:", nu_opt_kkt)

# 可视化部分
plt.figure(figsize=(10, 8))
contours = plt.contour(X1, X2, Z, 50)
plt.clabel(contours)
plt.plot(x_opt_qp[0], x_opt_qp[1], 'ro', label='QP Solution')
plt.plot(x_opt_kkt[0], x_opt_kkt[1], 'bo', label='KKT Solution')

# 可视化等式约束 x1 + x2 = 1
x1_eq = np.linspace(-1, 2, 100)
x2_eq = 1 - x1_eq
plt.plot(x1_eq, x2_eq, 'g--', label='$x_1 + x_2 = 1$')

# 可视化不等式约束 x1 >= 0 and x2 >= 0
plt.axvline(0, color='b', linestyle='--', label='$x_1 \geq 0$')
plt.axhline(0, color='y', linestyle='--', label='$x_2 \geq 0$')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('QP Problem and Solution with Constraints')
plt.legend()
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.show()
