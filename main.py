import numpy as np
import matplotlib.pyplot as plt
from all_fun import CEC2005
from bka import BKA
import matplotlib


matplotlib.rc("font", family='Microsoft YaHei')
# 参数设置
SearchAgents = 30  # 种群成员数量
Max_iterations = 300  # 最大迭代次数
number = 'F14'  # 选定优化函数，自行替换:F1~F23

# 加载函数数据 fobj, lb, ub, dim
fobj, lb, ub, dim = CEC2005(number)  # [lb,ub,D,y]：下界、上界、维度、目标函数表达式

# 调用算法
BKA_score, BKA_pos, BKA_Convergence_curve = BKA(SearchAgents, Max_iterations, [lb], [ub], dim, fobj)  # 调用BKA算法
print(BKA_score, BKA_pos)

# 绘制收敛曲线
CNT = 80
k = np.round(np.linspace(1, Max_iterations-1, CNT)).astype(int)  # 随机选CNT个点
iter_ = np.arange(1, Max_iterations + 1)
plt.subplot(1, 1, 1)
plt.plot(iter_[k], BKA_Convergence_curve[k], 'r->', linewidth=1)
plt.grid(True)
plt.title('收敛曲线')
plt.xlabel('迭代次数')
plt.ylabel('适应度值')
plt.legend(['BKA'])

plt.show()
