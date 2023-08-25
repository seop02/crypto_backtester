# Spin model을 이용한 주가 및 거래량 Monte Carlo simulation

#

# 참조 :

# [1] Bornholdt, 2001,

# Expectation bubbles in a spin model of matkets: Intermittency from frustration across scales.

#

# [2] Kaizoji, Bornholdt, et, al., 2002,

# Dynamics of price and trading volume in a spin model of stock markets with heterogeneous agents.

#

# Coding : 2020.09.02 아마추어 퀀트 (blog.naver.com/chunjein)

# ------------------------------------------------------------------------------------------------

import numpy as np

import numba as nb

import matplotlib.pyplot as plt



k = 100            # k * k개의 상태 격자

epochs = 3000      # 시뮬레이션 횟수

J = 1.0

alpha = 6

beta = 1 / 1.5   # T < Tc = 2.269 --> T = 1/beta

ns = k*k*3       # dt 동안 ns개의 spin을 업데이트 한다.

state_spin = np.random.choice([1,-1], [k, k])     # 초기 individual spin 값 (+1 or -1)

strategy_spin = np.random.choice([1,-1], [k, k])  # 초기의 individual strategy spin spin 값 (+1 or -1)

magnetization = []             # global magnetization

volume = []                    # 총 거래량 (매수량 + 매도량)

h = np.zeros([k, k])           # local field



@nb.jit(nopython=True)

def changeState_spin(state_spin, strategy_spin, h, mag):

    v = 0   # 거래량

    for i in range(ns):

        # 임의 지점의 state_spin를 선택해서 조건에 따라 state_spin 업데이트 한다.

        n = np.random.randint(0, k)

        m = np.random.randint(0, k)

        

        # 선택된 state_spin

        S = state_spin[n, m]

        C = strategy_spin[n, m]

        

        # S와 인접한 neighborhood state들의 합계

        nbstate = state_spin[(n + 1) % k, m] + state_spin[n, (m+1) % k] + state_spin[(n-1) % k, m] + state_spin[n,(m-1) % k]



        # local megnetization. [1]의 식 (2)

        h[n, m] = J * nbstate - alpha * C * mag

        

        # probability. [1]의 식 (1)

        p = 1 / (1 + np.exp(-2 * beta * h[n, m]))



        # change state_spin. [1]의 식 (1)

        if np.random.random() < p:

            state_spin[n, m] = +1

        else:

            state_spin[n, m] = -1

        

        # change strategy_spin. [1]의 식 (3)

        if alpha * S * C * mag < 0:

            strategy_spin[n, m] *= -1

        

        # 상태가 변화된 횟수를 기록한다. 상태 변화 -> 거래 발생 -> 거래량

        if S != state_spin[n, m]:

            v += 1

            

    return state_spin, strategy_spin, v / ns



# Ising Model Monte Calro simulation

# ----------------------------------

for t in range(epochs):

    magnetization.append(state_spin.mean()) # global magnetization : M(t)

    

    # agent들의 state_spin를 업데이트 한다.

    state_spin, strategy_spin, v = changeState_spin(state_spin, strategy_spin, h, magnetization[-1])

    volume.append(v)

    

    # agent들의 상태를 시각화한다.

    if t % 100 == 0:

        buy = str((state_spin > 0).sum())

        sell = str((state_spin < 0).sum())

        

        plt.figure(figsize=(4,4))

        plt.imshow(state_spin)

        plt.title("iteration = " + str(t) + ", buy = " + buy + ", sell = " + sell)

        plt.show()



# Stock price

skip = 10       # 초기 결과는 불안정하므로 이만큼 건너 뛴다.

mag = np.array(magnetization[skip:])



pstar = 10000   # 정상 가격

a = b = 1       # [2]의 a, b

n = k * k       # total spin 개수

m = (strategy_spin > 0).sum()      # fundamentalist 개수. [1]의 C(t)에 대한 설명

lam = b * n / (a * m)              # [2]의 식 (8)

price = pstar * np.exp(lam * mag)  # [2]의 식 (8)



# 주가 차트를 그린다.

plt.figure(figsize=(12, 5))

plt.plot(price[1:], linewidth=1.0, color='blue')

plt.title("Stock Price")

plt.show()



# 수익률 차트 (표준화)

rtn = np.diff(np.log(price))

norm_rtn = (rtn - rtn.mean()) / rtn.std()



plt.figure(figsize=(12, 4))

plt.plot(norm_rtn, linewidth=1.0, color='red')

plt.title("Return")

plt.axhline(y=0, linewidth=1.0, color='blue')

plt.show()



# 총 거래량 차트

V = volume[skip + 1:]

plt.figure(figsize=(12, 4))

plt.bar(np.arange(0, len(V)), V, color='green')

plt.title("Total Volume (buy + sell)")

plt.show()
