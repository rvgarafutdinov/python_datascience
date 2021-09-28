import pandas as pd
from requests import get
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.optimize._trustregion_constr.projections import projections

try:
    resp = get(r"https://raw.githubusercontent.com/rvgarafutdinov/python_datascience/master/task1_output.csv")
except Exception as e:
    print(f"Что-то пошло не так с загрузкой файла! А именно,\n{e}")
    exit(-1)

csv = StringIO(resp.text)  # будем работать со строкой как с файлом
df = pd.read_csv(csv, delimiter="|", header=None)  # получаем датафрейм

# теперь перезапишем все ячейки случайными данными
colscount = len(df.columns)
for irow in range(len(df)):
    df.iloc[irow, ] = np.random.randint(0, 100, colscount)

print(f"Датафрейм случайных чисел:\n{df.to_string()}")

# я не очень понял, как следует визуализировать в 3D мою таблицу 8x4
# (как-то далековат от трехмерной графики),
# поэтому построил график по точкам с координатами
# (x = 1 столбец, y = 2 столбец, z = поэл. произведение 3 и 4 столбцов)
x, y, z = df[0], df[1], df[2] * df[3]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, label="Кривая")
plt.show()
