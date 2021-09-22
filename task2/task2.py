import numpy as np
from scipy.linalg import solve
from tkinter import Tk, Entry, Label, Button, Frame, messagebox, filedialog, LEFT
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

proportion = [0.2] * 5  # вектор долей витаминов (глобальная переменная)
data = [[0] * 5] * 5  # данные по количеству витаминов


def buttonCalc_command(a, canvas, lbl):
    """Обработчик кнопки 'Рассчитать'"""
    global proportion, data
    try:
        data = [[float(y.get()) for y in x] for x in zip(*entriesMatrix)]
    except Exception as e:
        messagebox.showerror("Ошибка", f"Некорректный ввод единиц витаминов! Как следствие:\n{e}")
        return
    B = np.array([170, 180, 140, 180, 350]).reshape((5, 1))
    A = np.array(data)

    try:
        proportion = solve(A, B)
    except Exception as e:
        messagebox.showerror("Ошибка", f"При расчетах что-то пошло не так! А именно,\n{e}")
        return
    if any(c < 0 for c in proportion):
        messagebox.showwarning("Предупреждение",
                               "При текущих входных параметрах получаются отрицательные доли витаминов, "
                               "отрисовать это не получится ¯\_(ツ)_/¯")
        return

    a.clear()
    a.pie(proportion.flatten(), explode=explode, labels=labels, colors=colors,
          autopct="%1.1f%%", textprops={'fontsize': 7}, shadow=True, startangle=140)
    canvas.draw()

    setLabelflag(lbl, True)


def buttonSavedata_command():
    """Обработчик кнопки 'Сохранить данные'"""
    try:
        file = filedialog.asksaveasfile(mode='w', initialdir=".", filetypes=(("CSV files", "*.csv"),),
                                        initialfile="data", defaultextension=".png")
        np.savetxt(file.name, data, delimiter="|")
    except:
        pass


def buttonSavepie_command():
    """Обработчик кнопки 'Сохранить диаграмму'"""
    try:
        file = filedialog.asksaveasfile(mode='w', initialdir=".", filetypes=(("PNG files", "*.png"),),
                                        initialfile="figure", defaultextension=".png")
        fig.savefig(file.name)
    except:
        pass


def loadDefaultdata(matr, lbl):
    """Обработчик кнопки 'Вставить данные по умолчанию'"""
    global data
    data = [list(x) for x in zip([1, 9, 2, 1, 1],
                                 [10, 1, 2, 1, 1],
                                 [1, 0, 5, 1, 1],
                                 [2, 1, 1, 2, 9],
                                 [2, 1, 2, 13, 2])]
    for irow in range(5):
        for icell in range(5):
            matr[irow][icell].delete(0, "end")
            matr[irow][icell].insert(0, data[irow][icell])
    setLabelflag(lbl, False)


def setLabelflag(lbl, ok):
    """Установка флага необходимости пересчета"""
    if ok:
        lbl["text"] = "Рассчитано"
        lbl["fg"] = "green"
    else:
        lbl["text"] = "Нужен пересчет"
        lbl["fg"] = "red"


if __name__ == "__main__":
    root = Tk()
    root.resizable(width=False, height=False)
    root.title("Расчет витаминов")

    # фрейм для кнопок
    frameButtons = Frame(root)
    labelFlag = Label(frameButtons, text="Нужен пересчет", fg="red")

    # фрейм для полей ввода
    frameTable = Frame(root)
    frameTable.pack()
    labels = [f"Витамин {x}" for x in "ABCDE"]
    for icol, text in enumerate(["Продукт"] + labels):
        Label(frameTable, text=text).grid(row=0, column=icol, pady=2)
    for irow in range(1, 6):
        Label(frameTable, text=str(irow)).grid(row=irow, column=0, pady=2)
    entriesMatrix = []
    for irow in range(1, 6):
        entriesMatrix.append([])
        for icell in range(1, 6):
            entriesMatrix[-1].append(Entry(frameTable, width=12))
            entriesMatrix[-1][-1].bind("<KeyPress>", lambda key: setLabelflag(labelFlag, False))
            entriesMatrix[-1][-1].grid(row=irow, column=icell, pady=2)
            entriesMatrix[-1][-1].insert(0, 0)

    # фрейм для диаграммы
    framePie = Frame(root, width=30, height=100)
    framePie.pack(side=LEFT)

    fig = Figure(figsize=(3, 2))
    a = fig.add_subplot()
    sizes = proportion
    colors = "gold", "yellowgreen", "lightcoral", "lightskyblue"
    explode = 0, 0, 0.1, 0.1, 0
    a.pie(sizes, explode=explode, labels=labels, colors=colors,
          autopct="%1.1f%%", textprops={'fontsize': 7}, shadow=True, startangle=140)
    canvas = FigureCanvasTkAgg(fig, framePie)
    canvas.get_tk_widget().pack()
    canvas.draw()

    # фрейм для кнопок (продолжение)
    frameButtons.pack()
    buttonCalc = Button(frameButtons, text="Рассчитать", width=20)
    buttonCalc["command"] = lambda: buttonCalc_command(a, canvas, labelFlag)
    buttonCalc.pack()
    buttonSavedata = Button(frameButtons, text="Сохранить данные", width=20)
    buttonSavedata["command"] = buttonSavedata_command
    buttonSavedata.pack()
    buttonSavepie = Button(frameButtons, text="Сохранить диаграмму", width=20)
    buttonSavepie["command"] = buttonSavepie_command
    buttonSavepie.pack()
    labelFlag.pack()
    buttonDefaultdata = Button(frameButtons, text="Вставить данные по умолчанию", width=20, wraplength=100)
    buttonDefaultdata["command"] = lambda: loadDefaultdata(entriesMatrix, labelFlag)
    buttonDefaultdata.pack()

    root.mainloop()
