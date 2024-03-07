import numpy as np
trans = [
    [0.8, 0.4, 0.2],
    [0.2, 0.4, 0.6],
    [0, 0.2, 0.2]
]

states = [["s"], ["c"], ["r"]]
s_count, c_count, r_count = (0, 0, 0)


def sim(days):
    today = [[1], [0], [0]]
    num = days
    for i in range(2, num+1):
        tomorrow_p = np.dot(trans, today)
        tomorrow_d = np.random.choice(np.reshape(
            states, 3), replace=True, p=np.reshape(tomorrow_p, 3))
        if tomorrow_d == "s":
            today = np.array([[1], [0], [0]])
        elif tomorrow_d == "c":
            today = np.array([[0], [1], [0]])
        else:
            today = np.array([[0], [0], [1]])
    return tomorrow_d


for i in range(10000):
    wheather = sim(49)
    if wheather == "s":
        s_count = s_count+1
    elif wheather == "c":
        c_count = c_count+1
    else:
        r_count = r_count+1
stationary_distrubution = [s_count/10000, c_count/10000, r_count/10000]
print("Stationary Distriburion is" + str(stationary_distrubution))
