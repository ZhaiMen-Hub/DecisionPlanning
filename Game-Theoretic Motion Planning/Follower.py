import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
# N = 10
N = 3
Q = np.diag([0.0, 1.0, 2.0, 1.0, 2.0, 4.0])
R = np.diag([4.0, 4.0])
xref = np.array([0.0, 15.0, 0.0, 5.0, 0.0, 0.0])
x0 = np.array([2.0, 15.0, 0.0, 5.0, 0.0, 0.0])
tau = 1.0  # 时间步长，确保是浮点数

# 创建优化变量
X = ca.MX.sym('X', 6, N+1)
U = ca.MX.sym('U', 2, N)

# # 状态转移矩阵
# A_np = np.array([[1.0, tau, 0.5*tau**2, 0.0, 0.0, 0.0],
#                  [0.0, 1.0, tau, 0.0, 0.0, 0.0],
#                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                  [0.0, 0.0, 0.0, 1.0, tau, 0.5*tau**2],
#                  [0.0, 0.0, 0.0, 0.0, 1.0, tau],
#                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

# B_np = np.array([[1/6*tau**3, 0.0],
#                  [0.5*tau**2, 0.0],
#                  [tau, 0.0],
#                  [0.0, 1/6*tau**3],
#                  [0.0, 0.5*tau**2],
#                  [0.0, tau]])

# 状态转移矩阵
A_np = np.array([[1.0, tau, 0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, tau, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0, tau, 0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, tau],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

B_np = np.array([[0, 0.0],
                 [0, 0.0],
                 [tau, 0.0],
                 [0.0, 0],
                 [0.0, 0],
                 [0.0, tau]])

A = ca.MX(A_np)
B = ca.MX(B_np)

# 定义系统动力学
def dynamics(x, u):
    return ca.mtimes(A, x) + ca.mtimes(B, u)

# 定义优化问题
nx = 6  # 状态维度
nu = 2  # 输入维度
X_var = ca.MX.sym('X_var', nx, N+1)
U_var = ca.MX.sym('U_var', nu, N)

cost = 0
constraints = []

# 初始状态
constraints.append(X_var[:, 0] - x0)

# 轨迹规划的目标
for k in range(N):
    x_ref = ca.MX(xref)
    cost += ca.mtimes([(X_var[:, k] - x_ref).T, Q, (X_var[:, k] - x_ref)]) + ca.mtimes([U_var[:, k].T, R, U_var[:, k]])
    x_next = dynamics(X_var[:, k], U_var[:, k])
    constraints.append(X_var[:, k+1] - x_next)


# construct the optimization problem
nlp = {'x': ca.vertcat(ca.reshape(X_var, -1, 1), ca.reshape(U_var, -1, 1)), 'f': cost, 'g': ca.vertcat(*constraints)}

# 设置求解器
opts = {'ipopt': {'print_level': 0}}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# 求解优化问题
# sol = opti.solve()
# x0_var = np.concatenate((np.reshape(x0, (-1, 1)), np.zeros((nx * N, 1))), axis=0)
# u0_var = np.zeros((nu * N, 1))
# init_guess = np.concatenate((x0_var, u0_var), axis=0)
# sol = solver(x0=init_guess, lbx=-np.inf, ubx=np.inf, lbg=0, ubg=0)
# without initial guess
sol = solver(lbg=ca.DM(np.zeros((nx * (N+1), 1))), ubg=ca.DM(np.zeros((nx * (N+1), 1))))

# 提取解
sol_x = sol['x'].full().flatten()
X_sol = sol_x[:nx * (N+1)].reshape(N+1, nx).T
U_sol = sol_x[nx * (N+1):].reshape(N, nu).T
# print('X_sol')
# print(X_sol)
# print('U_sol')
# print(U_sol)

# 可视化结果
time = np.arange(N+1)
plt.figure(figsize=(14, 8))

# px 轨迹
plt.subplot(4, 2, 1)
plt.plot(time, X_sol[0, :], 'r-', label='px')
plt.plot(time, xref[0]*np.ones(N+1), 'b--', label='xref px')
plt.title('Position in x')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()

# py 轨迹
plt.subplot(4, 2, 2)
plt.plot(time, X_sol[3, :], 'r-', label='py')
plt.plot(time, xref[3]*np.ones(N+1), 'b--', label='xref py')
plt.title('Position in y')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()

# vx 轨迹
plt.subplot(4, 2, 3)
plt.plot(time, X_sol[1, :], 'r-', label='vx')
plt.legend()

# vy 轨迹
plt.subplot(4, 2, 4)
plt.plot(time, X_sol[4, :], 'r-', label='vy')
plt.legend()

# ax 轨迹
plt.subplot(4, 2, 5)
plt.plot(time, X_sol[2, :], 'r-', label='ax')
plt.legend()

# ay 轨迹
plt.subplot(4, 2, 6)
plt.plot(time, X_sol[5, :], 'r-', label='ay')
plt.legend()

# jx 轨迹
plt.subplot(4, 2, 7)
plt.plot(time[:-1], U_sol[0, :], 'g-', label='jx')
plt.title('Jerk in x')
plt.xlabel('Time step')
plt.ylabel('Jerk')
plt.legend()

# jy 轨迹
plt.subplot(4, 2, 8)
plt.plot(time[:-1], U_sol[1, :], 'g-', label='jy')
plt.title('Jerk in y')
plt.xlabel('Time step')
plt.ylabel('Jerk')
plt.legend()

# y-x 轨迹
plt.figure(figsize=(7, 5))
plt.plot(X_sol[0, :], X_sol[3, :], 'r-', label='Trajectory')
plt.title('Trajectory (y vs. x)')
plt.xlabel('px')
plt.ylabel('py')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()