from tkinter import *
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import pandas as pd
import sympy
from sympy.abc import x
import os

script_dir = os.getcwd()
data = pd.read_csv(os.path.normcase(os.path.join(script_dir, "SteelParameters.csv")))

def solve():
    for i in trv1.get_children():
        trv1.delete(i)
        
    graphBtn.config(state=NORMAL)
    data_array.clear()

    name = data['AISC_Manual_Label']
    flange_w2t = data['bf/2tf']
    ry = data['ry']
    j = data['J']
    Sx = data['Sx']
    ho = data['ho']
    rts = data['rts']
    Zx = data['Zx']
    Ix = data['Ix']
    weight = data['W']
    d = data['d']
    tw = data['tw']

    length_beam = float(length.get()) / 1000
    f_y = float(fy.get())
    left_Moment = -abs(float(leftMoment.get()))
    center_Moment = abs(float(centerMoment.get()))
    right_Moment = -abs(float(rightMoment.get()))
    left = np.abs(left_Moment) + center_Moment
    right = np.abs(right_Moment) + center_Moment
    left_Shear = abs(float(leftShear.get()))
    right_Shear = -abs(float(rightShear.get()))

    # Checking for compactness
    check_1 = 0.38*np.sqrt(200000/f_y)
    check_2 = np.sqrt(200000/f_y)

    compactness = np.zeros(len(flange_w2t), dtype="U25")
    compactness[flange_w2t <= check_1] = 'Compact'
    compactness[np.logical_and(flange_w2t > check_1, flange_w2t <= check_2)] = 'Noncompact'

    compact_logic = np.array(compactness == 'Compact')
    noncompact_logic = np.array(compactness == 'Noncompact')

    # For compact shape calculations
    Lp = np.zeros(compactness.shape[0])
    Lp = 1.76*ry*np.sqrt(200000/f_y)

    factor = j/(Sx*ho)
    Lr = np.zeros(compactness.shape[0])
    Lr = 1.95*rts*(200000/(0.7*f_y))*np.sqrt(factor + np.sqrt(np.power(factor, 2) + 6.76*np.power(0.7*f_y/200000, 2)))

    # Solution for Cb
    length_left = np.array(sympy.solve(np.power(length_beam, 2) - (2*length_beam*x) + (1-(right/left))*np.power(x, 2)))
    length_left = length_left[np.logical_and(length_left>0, length_left < length_beam)]
    slope = -left / np.power(length_left, 2)
    y_val = lambda x : slope*np.power(x-length_left, 2) + center_Moment
    Cb = 12.5*center_Moment / (2.5*center_Moment + 3*abs(y_val(length_beam/4)) + 4*abs(y_val(length_beam/2)) + 3*abs(y_val(3*length_beam/4)))

    # Design Strength
    logic_1 = length_beam <= Lp
    design_strength = np.zeros(len(flange_w2t))
    design_strength[np.logical_and(logic_1, compact_logic)] = 0.6*f_y*Zx[np.logical_and(logic_1, compact_logic)]

    logic_2 = np.logical_and(length_beam > Lp, length_beam <= Lr)
    Mp = 0.6*f_y*Zx
    design = Cb*(Mp[np.logical_and(logic_2, compact_logic)] - ((Mp[np.logical_and(logic_2, compact_logic)]-0.7*f_y*Sx[np.logical_and(logic_2, compact_logic)])*((length_beam-Lp[np.logical_and(logic_2, compact_logic)])/(Lr[np.logical_and(logic_2, compact_logic)]-Lp[np.logical_and(logic_2, compact_logic)]))))
    design_strength[np.logical_and(logic_2, compact_logic)] = 0.6*design[np.logical_and(logic_2, compact_logic)]

    logic_3 = length_beam > Lr
    Fcr = Cb*np.power(np.pi, 2)*200000 / np.power(length_beam/rts, 2) * np.sqrt(1 + 0.078*factor*np.power(length_beam/rts, 2))
    design_strength[np.logical_and(logic_3, compact_logic)] = 0.6*(Fcr*Sx)[np.logical_and(logic_3, compact_logic)]


    # For Noncompact shape calculations
    # Flange Local Buckling
    design_strength_1 = np.zeros(len(flange_w2t))
    design_strength_1[noncompact_logic] = Mp[noncompact_logic] - (Mp[noncompact_logic]-0.7*f_y*Sx[noncompact_logic])*((flange_w2t[noncompact_logic]-check_1)/(check_2-check_1))

    # Lateral-Torsional buckling:
    design_strength_2 = np.zeros(len(flange_w2t))
    design_strength_2[np.logical_and(logic_1, noncompact_logic)] = f_y*Zx[np.logical_and(logic_1, noncompact_logic)]
    design_strength_2[np.logical_and(logic_2, noncompact_logic)] = (Cb*(Mp - (Mp-0.7*f_y*Sx)*((length_beam-Lp)/(Lr-Lp))))[np.logical_and(logic_2, noncompact_logic)]
    design_strength_2[np.logical_and(logic_3, noncompact_logic)] = (Fcr*Sx)[np.logical_and(logic_3, noncompact_logic)]

    # Grabing the lowest value
    design = np.zeros(len(flange_w2t))
    des_1 = design_strength_1 < design_strength_2
    design[des_1] = design_strength_1[des_1]

    des_2 = design_strength_2 < design_strength_1
    design[des_2] = design_strength_1[des_2]

    design_strength[noncompact_logic] = 0.9*design[noncompact_logic]

    # Choosing Deisgn
    # Checking for moment
    max_deflection = float(deflection.get())
    
    inertia = center_Moment*((5*np.power(length_beam, 2)) / (48*200000*max_deflection))
    req_intertia = np.ones(Ix.shape[0]) * inertia
    max_moment_val = np.max(np.abs(np.array([left_Moment, center_Moment, right_Moment]))) * np.power(10, 3)
    max_moment = np.ones(Ix.shape[0]) * max_moment_val


    # Checking for shear
    shear = 0.4*f_y*d*tw
    max_shear_val = np.max(np.abs(np.array([left_Shear, right_Shear])))
    max_shear = np.ones(Ix.shape[0]) * max_shear_val


    trial_1 = np.logical_and(design_strength >= max_moment, (Ix*np.power(10, 6)) >= req_intertia)
    trial = np.logical_and(trial_1, shear > max_shear)
    beam_weight = (1/8)*(1.2*((weight*9.81)/1000))*np.power(length_beam/1000, 2) * 1000
    design_values = design_strength[trial]
    max_moment = (np.ones(beam_weight[trial].shape[0]) * max_moment_val) + (beam_weight[trial] * np.power(10, 6))
    allowable = design_values > max_moment[trial]


    name = data['AISC_Manual_Label'][trial][allowable]
    flange_w2t = data['bf/2tf'][trial][allowable]
    ry = data['ry'][trial][allowable]
    j = data['J'][trial][allowable]
    Sx = data['Sx'][trial][allowable]
    ho = data['ho'][trial][allowable]
    rts = data['rts'][trial][allowable]
    Zx = data['Zx'][trial][allowable]
    Ix = data['Ix'][trial][allowable]
    weight = data['W'][trial][allowable]
    d = data['d'][trial][allowable]
    tw = data['tw'][trial][allowable]
    check_1 = np.ones(data['tw'][trial][allowable].shape[0]) * (0.38*np.sqrt(200000/f_y))
    check_2 = np.ones(data['tw'][trial][allowable].shape[0]) * np.sqrt(200000/f_y)

    data_val = np.array([weight, name, compactness[trial][allowable], check_1, check_2, Lp[trial][allowable], Lr[trial][allowable],
        req_intertia[trial][allowable], max_moment[trial][allowable], 
        design_strength[trial][allowable], shear[trial][allowable], max_shear[trial][allowable]])
    data_val_sorted = data_val.T[np.argsort(data_val.T[:, 0])].T
    data_val = np.delete(data_val_sorted.T, [0], 1)
    data_array.append(data_val)

    table_val = np.array([name, flange_w2t, ry, j, Sx, ho, rts, Zx, Ix, weight, d, tw])
    sorted_val = table_val.T[np.argsort(table_val.T[:, 9])].T
    for i in range(sorted_val.shape[1]):
        array_dat = sorted_val.T[i].tolist()
        trv1.insert('', 'end', values=array_dat)


def graph():
    newWindow = Toplevel(mainWindow)
    newWindow.title("Steel Parameters")
    newWindow.geometry("500x690+800+0")
    newWindow.resizable(width=0, height=0)

    shearWrapper = LabelFrame(newWindow, text="Shear Diagram")
    momentWrapper = LabelFrame(newWindow, text="Moment Diagram")

    shearWrapper.place(x=10, y=10, width=480, height=335)
    momentWrapper.place(x=10, y=345, width=480, height=335)
    
    length_beam = float(length.get())

    def shear():
        left_Shear = abs(float(leftShear.get()))
        right_Shear = -abs(float(rightShear.get()))
        for widget in shearWrapper.winfo_children():
            widget.destroy()

        fig = Figure(figsize = (5, 10), dpi = 100)
        plot = fig.add_subplot(111)

        plot.plot([0, length_beam], [left_Shear, right_Shear], 'k')
        plot.plot([0, length_beam], [0, 0], 'k')
        plot.plot([0, 0], [0, left_Shear], 'k')
        plot.plot([length_beam, length_beam], [right_Shear, 0], 'k')

        canvas = FigureCanvasTkAgg(fig, master = shearWrapper)
        canvas.draw()
        canvas.get_tk_widget().place(x=10, y=10)
        toolbar = NavigationToolbar2Tk(canvas, shearWrapper)
        toolbar.update()
        canvas.get_tk_widget().pack()


    def moment():
        left_Moment = -abs(float(leftMoment.get()))
        center_Moment = abs(float(centerMoment.get()))
        right_Moment = -abs(float(rightMoment.get()))
        left = np.abs(left_Moment) + center_Moment
        right = np.abs(right_Moment) + center_Moment

        length_left = np.array(sympy.solve(np.power(length_beam, 2) - (2*length_beam*x) + (1-(right/left))*np.power(x, 2)))
        length_left = length_left[np.logical_and(length_left>0, length_left < length_beam)]

        slope = -left / np.power(length_left, 2)

        x_val = np.linspace(0, length_beam, int(length_beam))
        y_val = lambda x : slope*np.power(x-length_left, 2) + center_Moment

        for widget in momentWrapper.winfo_children():
            widget.destroy()

        fig = Figure(figsize = (5, 10), dpi = 100)
        plot = fig.add_subplot(111)

        plot.plot(x_val, y_val(x_val), 'k')
        plot.plot([0, length_beam], [0, 0], 'k')
        plot.plot([0, 0], [left_Moment, 0], 'k')
        plot.plot([length_beam, length_beam], [right_Moment, 0], 'k')

        canvas = FigureCanvasTkAgg(fig, master = momentWrapper)
        canvas.draw()
        canvas.get_tk_widget().place(x=10, y=10)
        toolbar = NavigationToolbar2Tk(canvas, momentWrapper)
        toolbar.update()
        canvas.get_tk_widget().pack()

    shear()
    moment()
    
def grab_val(event):
    item = trv1.item(trv1.focus())['values']
    data = data_array[0]

    values = data[data[:, 0] == item[0]][0]

    compact_text.config(text=f"The selected Steel is {values[1]}")
    check_1_text.config(text=f"λp = {np.round(values[2], 4)}")
    check_2_text.config(text=f"λr = {np.round(values[3], 4)}")
    Lp_text.config(text=f"Lp = {np.round(values[4], 4)}")
    Lr_text.config(text=f"Lr = {np.round(values[5], 4)}")
    design_strength_text.config(text=f"Allowable Strength = {np.round(values[8]/1000, 4)}")
    max_moment_text.config(text=f"Required Moment = {np.round(values[7]/1000, 4)}")
    required_inertia_text.config(text=f"Required Inertia = {np.round(values[6], 4)}")
    required_beam_shear.config(text=f"Required Shear = {np.round(values[-1], 4)}")
    beam_shear.config(text=f"Beam Shear = {np.round(values[-2] / 1000, 4)}")


mainWindow = Tk()
mainWindow.title("W-Shaped Steel Calculator (ASD - Metric Units)")
mainWindow.geometry("1148x690+0+0")
mainWindow.resizable(width=0, height=0)

wrapper1 = LabelFrame(mainWindow, text="Beam Properties")
wrapper2 = LabelFrame(mainWindow, text="Table of Usable W-Shaped Steel")
wrapper3 = LabelFrame(mainWindow, text="Solved values of selected steel")

wrapper1.place(x=10, y=10, width=380, height=325)
wrapper2.place(x=10, y=345, width=1128, height=335)
wrapper3.place(x=410, y=10, width=728, height=325)


# Variables
data_array = []
asd_data = []
leftMoment = StringVar()
centerMoment = StringVar()
rightMoment = StringVar()
leftShear = StringVar()
rightShear = StringVar()
length = StringVar()
fy = StringVar()
deflection = StringVar()


# Wrapper 1
Label(wrapper1, text="Left Moment:").place(x=10, y=10)
Label(wrapper1, text="Center Moment:").place(x=10, y=40)
Label(wrapper1, text="Right Moment:").place(x=10, y=70)
Label(wrapper1, text="Left Shear:").place(x=10, y=100)
Label(wrapper1, text="Right Shear:").place(x=10, y=130)
Label(wrapper1, text="Length:").place(x=10, y=160)
Label(wrapper1, text="Fy:").place(x=10, y=190)
Label(wrapper1, text="Max Deflection:").place(x=10, y=220)

Entry(wrapper1, textvariable=leftMoment).place(x=130, y=10)
Entry(wrapper1, textvariable=centerMoment).place(x=130, y=40)
Entry(wrapper1, textvariable=rightMoment).place(x=130, y=70)
Entry(wrapper1, textvariable=leftShear).place(x=130, y=100)
Entry(wrapper1, textvariable=rightShear).place(x=130, y=130)
Entry(wrapper1, textvariable=length).place(x=130, y=160)
Entry(wrapper1, textvariable=fy).place(x=130, y=190)
Entry(wrapper1, textvariable=deflection).place(x=130, y=220)

Label(wrapper1, text="kN-m").place(x=260, y=10)
Label(wrapper1, text="kN-m").place(x=260, y=40)
Label(wrapper1, text="kN-m").place(x=260, y=70)
Label(wrapper1, text="kN").place(x=260, y=100)
Label(wrapper1, text="kN").place(x=260, y=130)
Label(wrapper1, text="mm").place(x=260, y=160)
Label(wrapper1, text="MPa").place(x=260, y=190)
Label(wrapper1, text="mm").place(x=260, y=220)


solveBtn = Button(wrapper1, text="Solve", command=solve)
solveBtn.place(x=30, y=260, width=100)

graphBtn = Button(wrapper1, text="Show Graph", command=graph, state=DISABLED)
graphBtn.place(x=150, y=260, width=100)


# Wrapper 2
trv1 = ttk.Treeview(wrapper2, columns=(1,2,3,4,5,6,7,8,9,10,11,12), show="headings", height=14)
trv1.place(x=5, y=5, width=1095)
trv1.heading(1, text='AISC_Manual_Label')
trv1.heading(2, text='bf/2tf')
trv1.heading(3, text='ry')
trv1.heading(4, text='J')
trv1.heading(5, text='Sx')
trv1.heading(6, text='ho')
trv1.heading(7, text='rts')
trv1.heading(8, text='Zx')
trv1.heading(9, text='Ix')
trv1.heading(10, text='W')
trv1.heading(11, text='d')
trv1.heading(12, text='tw')
trv1.column(1, width=135, anchor="c")
trv1.column(2, width=87, anchor="c")
trv1.column(3, width=87, anchor="c")
trv1.column(4, width=87, anchor="c")
trv1.column(5, width=87, anchor="c")
trv1.column(6, width=89, anchor="c")
trv1.column(7, width=87, anchor="c")
trv1.column(8, width=87, anchor="c")
trv1.column(9, width=87, anchor="c")
trv1.column(10, width=87, anchor="c")
trv1.column(11, width=87, anchor="c")
trv1.column(12, width=87, anchor="c")
vsb1 = ttk.Scrollbar(wrapper2, orient="vertical", command=trv1.yview)
vsb1.place(x=1101, y=6, height=304)
trv1.configure(yscrollcommand=vsb1.set)
trv1.bind("<<TreeviewSelect>>", grab_val)


# Wrapper 3
compact_text = Label(wrapper3, text="")
compact_text.place(x=10, y=10)

check_1_text = Label(wrapper3, text="")
check_1_text.place(x=10, y=40)

check_2_text = Label(wrapper3, text="")
check_2_text.place(x=10, y=70)

Lp_text = Label(wrapper3, text="")
Lp_text.place(x=10, y=100)

Lr_text = Label(wrapper3, text="")
Lr_text.place(x=10, y=130)

required_inertia_text = Label(wrapper3, text="")
required_inertia_text.place(x=10, y=160)

design_strength_text = Label(wrapper3, text="")
design_strength_text.place(x=10, y=190)

max_moment_text = Label(wrapper3, text="")
max_moment_text.place(x=10, y=220)

required_beam_shear = Label(wrapper3, text="")
required_beam_shear.place(x=10, y=250)

beam_shear = Label(wrapper3, text="")
beam_shear.place(x=10, y=280)

mainWindow.mainloop()




# λλλλλλλλλλ