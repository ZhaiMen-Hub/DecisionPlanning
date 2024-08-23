import casadi as ca
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

current_file_path = os.path.dirname(os.path.abspath(__file__))

# 参数设置
T = 10
tau = 0.5
N = int(T/tau)  # planning horizon & planning times
# tau = 1
# N = 5
Q = np.diag([0.0, 1.0, 2.0, 1.0, 2.0, 4.0])
R = np.diag([4.0, 4.0])
xF0 = np.array([2.0, 15.0, 0.0, 5.0, 0.0, 0.0])
xFref = np.array([0.0, 15.0, 0.0, 5.0, 0.0, 0.0])
xL0 = np.array([32.0, 5.0, 0.0, 3.0, 0.0, 0.0])
xLref = np.array([0.0, 5.0, 0.0, 5.0, 0.0, 0.0])
addCollisionCons = True
KF = 0.01
KL = 1 - KF
Kinfluence = 0
tolerance = 5e-2
accThreshold = 0.4  # Dead zone threshold of accelation for Accel / Dccel prediction
probUpperBound = 0.95 # maximum probility
probGrid = 0.1

distL = 15
# distF = 10
# distF = 20
distDecelF = 20    # dist for follower to decel
distAccelF = 10
# decel = True        # real prob
decel = False
# probDecel0 = 0.99   # init prob 
# probDecel0 = 0.01   # init prob 
probDecel0 = 0.5
probAccel0 = 1 - probDecel0

# KF = 0.5
# KL = 1 - KF
# distF = 20    # collision ditance
# distL = 20
# Kinfluence = 1    # enable Jinfluence

# 状态转移矩阵
A_np = np.array([[1.0, tau, 0.5*tau**2, 0.0, 0.0, 0.0],
                 [0.0, 1.0, tau, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0, tau, 0.5*tau**2],
                 [0.0, 0.0, 0.0, 0.0, 1.0, tau],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

B_np = np.array([[1/6*tau**3, 0.0],
                 [0.5*tau**2, 0.0],
                 [tau, 0.0],
                 [0.0, 1/6*tau**3],
                 [0.0, 0.5*tau**2],
                 [0.0, tau]])

A = ca.MX(A_np)
B = ca.MX(B_np)

# 定义系统动力学
def dynamics(x, u):
    return ca.mtimes(A, x) + ca.mtimes(B, u)

# Function for Leader Follower Game with Deceleration and Acceleration
def gameLeaderFollower(xL0, xF0, probDecel, probAccel):

    # 定义优化问题
    nx = 6  # 状态维度
    nu = 2  # 输入维度
    lenX = nx * (N+1)
    lenU = nu * N

    XDecelF = ca.MX.sym('XDecelF', nx, N+1)
    UDecelF = ca.MX.sym('UDecelF', nu, N)
    XDecelL = ca.MX.sym('XDecelL', nx, N+1)
    UDecelL = ca.MX.sym('UDecelL', nu, N)
    XAccelF = ca.MX.sym('XAccelF', nx, N+1)
    UAccelF = ca.MX.sym('UAccelF', nu, N)
    XAccelL = ca.MX.sym('XAccelL', nx, N+1)
    UAccelL = ca.MX.sym('UAccelL', nu, N)

    # Initialize costs and constraints
    JDecelF = 0
    JDecelL = 0
    JAccelF = 0
    JAccelL = 0
    consDecelF = []
    consDecelL = []
    consAccelF = []
    consAccelL = []
    collisionConsDecelF = []
    collisionConsDecelL = []
    collisionConsAccelF = []
    collisionConsAccelL = []

    # Initial state constraints
    consDecelF.append(XDecelF[:, 0] - xF0)
    consDecelL.append(XDecelL[:, 0] - xL0)
    consAccelF.append(XAccelF[:, 0] - xF0)
    consAccelL.append(XAccelL[:, 0] - xL0)

    # Define reference trajectories
    xFrefCa = ca.MX(xFref)
    xLrefCa = ca.MX(xLref)

    for k in range(N):
        # Cost function for deceleration and acceleration
        JDecelF += ca.mtimes([(XDecelF[:, k+1] - xFrefCa).T, Q, (XDecelF[:, k+1] - xFrefCa)]) + ca.mtimes([UDecelF[:, k].T, R, UDecelF[:, k]])
        JDecelL += ca.mtimes([(XDecelL[:, k+1] - xLrefCa).T, Q, (XDecelL[:, k+1] - xLrefCa)]) + ca.mtimes([UDecelL[:, k].T, R, UDecelL[:, k]])
        JAccelF += ca.mtimes([(XAccelF[:, k+1] - xFrefCa).T, Q, (XAccelF[:, k+1] - xFrefCa)]) + ca.mtimes([UAccelF[:, k].T, R, UAccelF[:, k]])
        JAccelL += ca.mtimes([(XAccelL[:, k+1] - xLrefCa).T, Q, (XAccelL[:, k+1] - xLrefCa)]) + ca.mtimes([UAccelL[:, k].T, R, UAccelL[:, k]])

        # Dynamics constraints
        xF_next_dec = dynamics(XDecelF[:, k], UDecelF[:, k])
        xL_next_dec = dynamics(XDecelL[:, k], UDecelL[:, k])
        xF_next_acc = dynamics(XAccelF[:, k], UAccelF[:, k])
        xL_next_acc = dynamics(XAccelL[:, k], UAccelL[:, k])

        consDecelF.append(XDecelF[:, k+1] - xF_next_dec)
        consDecelL.append(XDecelL[:, k+1] - xL_next_dec)
        consAccelF.append(XAccelF[:, k+1] - xF_next_acc)
        consAccelL.append(XAccelL[:, k+1] - xL_next_acc)

        # Collision constraints
        if addCollisionCons:
            collisionConsDecelF.append(distDecelF + XDecelF[0, k+1] - XDecelL[0, k+1])
            collisionConsDecelL.append(distL + XDecelF[0, k+1] - XDecelL[0, k+1])
            collisionConsAccelF.append(distAccelF + XAccelF[0, k+1] - XAccelL[0, k+1])
            collisionConsAccelL.append(distL + XAccelF[0, k+1] - XAccelL[0, k+1])

    # Combine constraints
    equConDecelF = ca.vertcat(*consDecelF)
    equConDecelL = ca.vertcat(*consDecelL)
    equConAccelF = ca.vertcat(*consAccelF)
    equConAccelL = ca.vertcat(*consAccelL)
    inequConsDecelF = ca.vertcat(*collisionConsDecelF)
    inequConsDecelL = ca.vertcat(*collisionConsDecelL)
    inequConsAccelF = ca.vertcat(*collisionConsAccelF)
    inequConsAccelL = ca.vertcat(*collisionConsAccelL)
    inequCon = ca.veccat(inequConsDecelF, inequConsDecelL, inequConsAccelF, inequConsAccelL)

    # Define Lagrangian multipliers
    lambda_dec = ca.MX.sym('lambda_dec', equConDecelF.size1())
    lambda_acc = ca.MX.sym('lambda_acc', equConAccelF.size1())
    mu_dec = ca.MX.sym('mu_dec', inequConsDecelF.size1())
    mu_acc = ca.MX.sym('mu_acc', inequConsAccelF.size1())

    # Lagrangian for deceleration and acceleration
    LagrangianDecel = JDecelF + \
                      ca.mtimes([lambda_dec.T, equConDecelF]) + \
                      ca.mtimes([mu_dec.T, inequConsDecelF])
    LagrangianAccel = JAccelF + \
                      ca.mtimes([lambda_acc.T, equConAccelF]) + \
                      ca.mtimes([mu_acc.T, inequConsAccelF])

    # KKT conditions
    grad_L_x_decel = ca.gradient(LagrangianDecel, ca.vertcat(ca.reshape(XDecelF, -1, 1), ca.reshape(UDecelF, -1, 1)))
    grad_L_x_accel = ca.gradient(LagrangianAccel, ca.vertcat(ca.reshape(XAccelF, -1, 1), ca.reshape(UAccelF, -1, 1)))
    complementary_slackness_decel = ca.diag(mu_dec) @ inequConsDecelF
    complementary_slackness_accel = ca.diag(mu_acc) @ inequConsAccelF

    # Contingency MPC
    CMpcCon = UDecelL[:, 0] - UAccelL[:, 0]

    # equality constraints
    equCon = ca.vertcat(equConDecelF, equConAccelF, equConDecelL, equConAccelL,
                        grad_L_x_decel, grad_L_x_accel, complementary_slackness_decel, complementary_slackness_accel, CMpcCon)

    # Construct the optimization problem
    x = ca.vertcat(ca.reshape(XDecelF, -1, 1), ca.reshape(UDecelF, -1, 1),
                   ca.reshape(XDecelL, -1, 1), ca.reshape(UDecelL, -1, 1),
                   ca.reshape(XAccelF, -1, 1), ca.reshape(UAccelF, -1, 1),
                   ca.reshape(XAccelL, -1, 1), ca.reshape(UAccelL, -1, 1), lambda_dec, lambda_acc, mu_dec, mu_acc)

    # Lower bound to x (add constraint mu >= 0)
    lbx = ca.vertcat(
        ca.repmat(-ca.inf, 4 * (lenU + lenX) + lambda_dec.size1() + lambda_acc.size1(), 1),  # 生成全为 -ca.inf 的列向量
        ca.repmat(0, mu_dec.size1() + mu_acc.size1(), 1)                                  # 生成全为 0 的列向量(for mu)
        # ca.repmat(-tolerance, mu_dec.size1() + mu_acc.size1(), 1)                                  # 生成全为 0 的列向量(for mu)
    )

    # NLP formulation
    nlp = {'x': x, 'f': probDecel * (KF * JDecelF + KL * JDecelL) + probAccel * (KF * JAccelF + KL * JAccelL) , 'g': ca.vertcat(equCon, inequCon)}

    # Solver options
    opts = {'ipopt': {'print_level': 0}}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve the optimization problem
    sol = solver(
        lbg=ca.vertcat(
            # ca.repmat(0, equCon.size1(), 1),          # 全为 0 的列向量(for equality)
            ca.repmat(-tolerance, equCon.size1(), 1),          # 全为 0 的列向量(for equality)
            ca.repmat(-ca.inf, inequCon.size1(), 1)   # 全为 -ca.inf 的列向量(for inequality, <=0 )
        ),
        ubg=ca.repmat(0, equCon.size1() + inequCon.size1(), 1),  # 全为 0 的列向量
        # ubg=ca.repmat(tolerance, equCon.size1() + inequCon.size1(), 1),  # 全为 0 的列向量
        lbx=lbx
    )

    # Extract solutions
    sol_x = sol['x'].full().flatten()
    XDecelF_sol = sol_x[:lenX].reshape(N+1, nx).T
    UDecelF_sol = sol_x[lenX:lenX+lenU].reshape(N, nu).T
    XDecelL_sol = sol_x[lenX+lenU:2*lenX+lenU].reshape(N+1, nx).T
    UDecelL_sol = sol_x[2*lenX+lenU:2*lenX+2*lenU].reshape(N, nu).T
    XAccelF_sol = sol_x[2*lenX+2*lenU:3*lenX+2*lenU].reshape(N+1, nx).T
    UAccelF_sol = sol_x[3*lenX+2*lenU:3*lenX+3*lenU].reshape(N, nu).T
    XAccelL_sol = sol_x[3*lenX+3*lenU:4*lenX+3*lenU].reshape(N+1, nx).T
    UAccelL_sol = sol_x[4*lenX+3*lenU:4*lenX+4*lenU].reshape(N, nu).T

    return XDecelF_sol, UDecelF_sol, XDecelL_sol, UDecelL_sol, XAccelF_sol, UAccelF_sol, XAccelL_sol, UAccelL_sol


# Initialize lists to store the results for animation and tracking
XDecelF_hist = []
UDecelF_hist = []
XDecelL_hist = []
UDecelL_hist = []
XAccelF_hist = []
UAccelF_hist = []
XAccelL_hist = []
UAccelL_hist = []

XF_actual = []
UF_actual = []
XL_actual = []
UL_actual = []

probDecel_hist = []
probAccel_hist = []

# Initial conditions for rolling horizon
xF_curr = xF0
xL_curr = xL0
probDecel_curr = probDecel0
probAccel_curr = probAccel0

# Perform rolling horizon optimization
for n in range(N):

    # Solve the optimization problem using gameLeaderFollower
    XDecelF_sol, UDecelF_sol, XDecelL_sol, UDecelL_sol, XAccelF_sol, UAccelF_sol, XAccelL_sol, UAccelL_sol = gameLeaderFollower(xL_curr, xF_curr, probDecel_curr, probAccel_curr)
    
    # Store the planned trajectories and controls
    XDecelF_hist.append(XDecelF_sol)
    UDecelF_hist.append(UDecelF_sol)
    XDecelL_hist.append(XDecelL_sol)
    UDecelL_hist.append(UDecelL_sol)
    XAccelF_hist.append(XAccelF_sol)
    UAccelF_hist.append(UAccelF_sol)
    XAccelL_hist.append(XAccelL_sol)
    UAccelL_hist.append(UAccelL_sol)
    probDecel_hist.append(probDecel_curr)
    probAccel_hist.append(probAccel_curr)

    # Record the actual state and control executed
    XL_actual.append(xL_curr)
    XF_actual.append(xF_curr)
    UL_actual.append(UDecelL_sol[:, 0])  # UDecelL_sol[:, 0] == UAccelL_sol[:, 0]
    if decel:
        UF_actual.append(UDecelF_sol[:, 0])
    else:
        UF_actual.append(UAccelF_sol[:, 0])

    # Update Decel and Accel probability (when neither prob achieve upper bound)
    # n == 0 corespond to init acceleration
    if n != 0 and probDecel_curr < probUpperBound and probAccel_curr < probUpperBound:
        if xF_curr[2] < -accThreshold:
            probDecel_curr = min(probDecel_curr + probGrid, probUpperBound)  # predict that the follower will decel
            probAccel_curr = 1 - probDecel_curr
        else:
            probAccel_curr = min(probAccel_curr + probGrid, probUpperBound)  # predict that the follower will accel
            probDecel_curr = 1 - probAccel_curr
    
    # Update initial conditions for the next iteration based on executed control
    xL_curr = XDecelL_sol[:, 1] # Update leader state to the next step
    # Update follower state to the next step
    if decel:
        xF_curr = XDecelF_sol[:, 1]
    else:
        xF_curr = XAccelF_sol[:, 1]


# plotting

np.set_printoptions(precision=2)
# print('XF_sol')
# print(XF_sol)
# print('UF_sol')
# print(UF_sol)

# # 可视化结果
# time = np.arange(N+1)
# plt.figure(figsize=(14, 8))

# # px 轨迹
# plt.subplot(4, 2, 1)
# plt.plot(time, XF_sol[0, :], 'r-', label='px')
# plt.plot(time, xFref[0]*np.ones(N+1), 'b--', label='xref px')
# plt.title('Position in x')
# plt.xlabel('Time step')
# plt.ylabel('Position')
# plt.legend()

# # py 轨迹
# plt.subplot(4, 2, 2)
# plt.plot(time, XF_sol[3, :], 'r-', label='py')
# plt.plot(time, xFref[3]*np.ones(N+1), 'b--', label='xref py')
# plt.title('Position in y')
# plt.xlabel('Time step')
# plt.ylabel('Position')
# plt.legend()

# # vx 轨迹
# plt.subplot(4, 2, 3)
# plt.plot(time, XF_sol[1, :], 'r-', label='vx')
# plt.legend()

# # vy 轨迹
# plt.subplot(4, 2, 4)
# plt.plot(time, XF_sol[4, :], 'r-', label='vy')
# plt.legend()

# # ax 轨迹
# plt.subplot(4, 2, 5)
# plt.plot(time, XF_sol[2, :], 'r-', label='ax')
# plt.legend()

# # ay 轨迹
# plt.subplot(4, 2, 6)
# plt.plot(time, XF_sol[5, :], 'r-', label='ay')
# plt.legend()

# # jx 轨迹
# plt.subplot(4, 2, 7)
# plt.plot(time[:-1], U_sol[0, :], 'g-', label='jx')
# plt.title('Jerk in x')
# plt.xlabel('Time step')
# plt.ylabel('Jerk')
# plt.legend()

# # jy 轨迹
# plt.subplot(4, 2, 8)
# plt.plot(time[:-1], U_sol[1, :], 'g-', label='jy')
# plt.title('Jerk in y')
# plt.xlabel('Time step')
# plt.ylabel('Jerk')
# plt.legend()

# # y-x 轨迹
# plt.figure(figsize=(7, 5))
# plt.plot(XF_sol[0, :], XF_sol[3, :], 'r-', label='Trajectory')
# plt.title('Trajectory (y vs. x)')
# plt.xlabel('px')
# plt.ylabel('py')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# # plt.show()

# Create the animation
# fig, ax = plt.subplots(3, 1, figsize=(10, 8))
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plt.tight_layout()

# data reforming
XF_actual = np.array(XF_actual).T # 6 * N
XL_actual = np.array(XL_actual).T

def update(frame):

    # clear plotting of last timestep
    ax[0].clear()
    ax[1].clear()
    
    # plot traj
    ax[0].plot(XF_actual[0, : frame+1], XF_actual[3, : frame+1], 'bo-', label='Traj_F', linewidth=1)
    ax[0].plot(XL_actual[0, : frame+1], XL_actual[3, : frame+1], 'mo-', label='Traj_L', linewidth=1)

    # plot contingencyplan
    # decel
    ax[0].plot(XDecelL_hist[frame][0, :], XDecelL_hist[frame][3, :], 'r', label='dec', linewidth=3, alpha=probDecel_hist[frame])
    # accel
    ax[0].plot(XAccelL_hist[frame][0, :], XAccelL_hist[frame][3, :], 'r', label='acc', linewidth=3, alpha=probAccel_hist[frame])

    # # plot collision distance
    # ax[0].axvline(x = XF_actual[0, frame] + distF, color='b', linestyle='--', label='collisionF')
    # ax[0].axvline(x = XL_actual[0, frame] - distL, color='r', linestyle='--', label='collisionL')

    # Add labels
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend(loc='lower right')
    # ax[0].set_title('State Evolution')
    ax[0].set_title(f't = {frame * tau}, Pd = {probDecel_hist[frame]:.2f}, Pa = {probAccel_hist[frame]:.2f}')
    
    # Set limit
    ax[0].set_xlim(0, 160)
    ax[0].set_ylim(2, 6)
    ax[0].grid(True)

    # Plot ax[1] vel - time
    time = np.arange(0, frame * tau + tau, tau)
    ax[1].plot(time, XF_actual[1, : frame+1], 'bo-', label='vxF')
    ax[1].plot(time, XL_actual[1, : frame+1], 'mo-', label='vxL')
    
    # # Add labels
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('vx')
    ax[1].legend()
    # ax[1].set_title('Control Inputs')

    # # Set limit
    ax[1].set_xlim(0, T+2*tau)
    ax[1].set_ylim(4, 18)
    ax[1].grid(True)

    # # plot ax[2] Acc - time
    # ax[2].clear()
    # ax[2].plot(time, XF_actual[2, : frame+1], 'bo-', label='axF')
    # ax[2].plot(time, XL_actual[2, : frame+1], 'mo-', label='axL')
    # # ax[2].plot(time, XL_actual[2, : frame+1] - XF_actual[2, : frame+1], 'ro-', label='axD')
    
    # # # Add labels
    # ax[2].set_xlabel('t')
    # ax[2].set_ylabel('ax')
    # ax[2].legend()
    # # ax[1].set_title('Control Inputs')

    # # # Set limit
    # ax[2].set_xlim(0, T+2*tau)
    # # ax[2].set_ylim(-2, 2)
    # ax[2].grid(True)

    # save fig
    plt.savefig(f'{current_file_path}/log/CMPCGame_{frame}.jpg')

ani = FuncAnimation(fig, update, frames=range(N), repeat=False)
ani.save(f'{current_file_path}/log/CMPCGame_animation.gif', writer='pillow')
ani.save(f'{current_file_path}/log/CMPCGame_animation.mp4', writer='ffmpeg')

# plt.show()