import numpy as np
trans = [
    [0.8, 0.4, 0.2],
    [0.2, 0.4, 0.6],
    [0, 0.2, 0.2]
]
today = [[0], [0], [1]]
states = [["s"], ["c"], ["r"]]
num = int(input("How many days: "))
if today[0][0] == 1:
    print("day 1 is sunny")
elif today[1][0] == 1:
    print("day 1 is cloudy")
else:
    print("day 1 is rainy")

for i in range(2, num+1):
    tormorrow_p = np.dot(trans, today)
    tomorrow_d = np.random.choice(np.reshape(
        states, 3), replace=True, p=np.reshape(tormorrow_p, 3))
    if tomorrow_d == "s":
        print("day {} probable is sunny".format(i))
        today = np.array([[1], [0], [0]])
    elif tomorrow_d == "c":
        print("day {} probable is cloudy".format(i))
        today = np.array([[0], [1], [0]])
    else:
        print("day {} probable is rainy".format(i))
        today = np.array([[0], [0], [1]])
