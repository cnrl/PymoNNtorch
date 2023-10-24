import matplotlib.pyplot as plt
import numpy as np
import csv

ekw = dict(ecolor=(0, 0, 0, 1.0), lw=1, capsize=3, capthick=1)

color1 = (0, 176/255, 80/255, 1)
color2 = (1/255, 127/255, 157/255, 1)
color3 = (120/255,110/255,120/255,1)#(117/255,117/255,117/255,1)#(253/255, 97/255, 0, 1)

#color11 = (0, 176/255, 80/255, 0.7)
color11 = (0, 176/255*0.5, 80/255*0.5, 0.7)
#color11 = (0, 0, 0, 1.0)

lookup = {'brian_LIF.py': ['Brian2', color2],
          'brian_LIF_cpp.py': ['Brian2 C++', color2],
          'brian_LIF_gpu.py': ['Brian2 GPU', color2],
          'nest_native_LIF.py': ['Nest', color3],
          'pymonnto_fast_LIF.py': ['PymoNNto', color11],
          'pymonnto_slow_LIF.py': ['PymoNNto', color11],# naive
          'pymonntorch_fast_LIF_cpu.py': ['Pymonntorch CPU', color1],
          'pymonntorch_fast_LIF_cuda.py': ['Pymonntorch GPU', color1],
          'pymonntorch_slow_LIF_cpu.py': ['Pymonntorch CPU', color1],# naive
          'pymonntorch_slow_LIF_cuda.py': ['Pymonntorch GPU', color1],# naive

          'brian_izh.py': ['Brian2', color2],
          'brian_izh_cpp.py': ['Brian2 C++', color2],
          'brian_izh_cuda.py': ['Brian2 GPU', color2],
          'pymonnto_izh.py': ['PymoNNto', color11],
          'pymonntorch_izh_cpu.py': ['PymoNNtorch CPU', color1],
          'pymonntorch_izh_cuda.py': ['PymoNNtorch GPU', color1],
          'pynn_nest_izh.py': ['Nest (PyNN)', color3]
          }

def load(filename):
    sim_col = []
    measurements = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i==0:
                sim_col = [lookup[s] for s in row[1:]]
            else:
                measurements.append([float(s) for s in row])

    return sim_col, np.array(measurements)[:,1:] #remove enumeration in first column


fig, ax = plt.subplots(1, 1)
fig.set_figwidth(4)#12
fig.set_figheight(4)

axis = ax#ax[2]
sim_col, data = load('Results/Swift-SF315-51G/Simple2.csv')
measurements = np.mean(data, axis=0)
err = np.std(data, axis=0)
simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
index = list(range(len(sim_col)))
index = [1,2,3,5,6,4]#5,6,7, 2,3,1

text_gap = np.max(measurements)*0.01
for i, s, m, e, c in zip(index, simulators, measurements, err, colors):
    axis.bar(i, m, width=0.8, color=c, yerr=e, error_kw=ekw)
    axis.text(i, m+e+text_gap, '{0:.2f}'.format(m)+'s', ha='center', va='bottom', color=c)  # , color='gray'

axis.tick_params(axis='both', which='both', length=0)
axis.set_yticks([], [])
axis.set_ylim([0, np.max(measurements)*1.5])#1.3
axis.set_xticks(np.array(index), simulators, rotation=30, ha="right")
axis.spines[['left', 'right', 'top']].set_visible(False)
axis.set_title('Simple LIF with\nOne Step STDP', fontweight='bold') #Simple LIF with\nOne Step STDP\n optimized vs naive

for xtick, color in zip(axis.get_xticklabels(), colors):
    xtick.set_color(color)

axis.plot([3.5, 3.5], [0.0, 60], '--', c='black', linewidth=1)

axis.text(2, 55, 'optimized', size=12, ha='center', va='bottom', color=(0,0,0,1), fontweight='bold')
axis.text(5, 55, 'naive', size=12, ha='center', va='bottom', color=(0,0,0,1), fontweight='bold')

fig.tight_layout()
#plt.savefig('filename.png', dpi=600)
plt.show()





fig, ax = plt.subplots(1, 2)
fig.set_figwidth(8)#12
fig.set_figheight(4)


axis = ax[0]
sim_col, data = load('Results/Swift-SF315-51G/Simple.csv')
measurements = np.mean(data, axis=0)
err = np.std(data, axis=0)
simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
#index = list(range(len(sim_col)))
index = [3,4,2,1,5,6,7]

text_gap = np.max(measurements)*0.01
for i, s, m, e, c in zip(index, simulators, measurements, err, colors):
    axis.bar(i, m, width=0.8, color=c, yerr=e, error_kw=ekw)
    axis.text(i, m + e+text_gap, '{0:.2f}'.format(m)+'s', ha='center', va='bottom', color=c)


axis.tick_params(axis='both', which='both', length=0)
axis.set_yticks([], [])
axis.set_ylim([0, np.max(measurements)*1.3])
axis.set_xticks(index, simulators, rotation=30, ha="right")
axis.spines[['left', 'right', 'top']].set_visible(False)
axis.set_title('Simple LIF with\nOne Step STDP', fontweight='bold')

for xtick, color in zip(axis.get_xticklabels(), colors):
    xtick.set_color(color)







axis = ax[1]
sim_col, data = load('Results/Swift-SF315-51G/Izhikevich.csv')
measurements = np.mean(data, axis=0)
err = np.std(data, axis=0)
simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
#index = list(range(len(sim_col)))
index = [4,3,2,5,6,7,1]

text_gap = np.max(measurements)*0.01
for i, s, m, e, c in zip(index, simulators, measurements, err, colors):
    axis.bar(i, m, width=0.8, color=c, yerr=e, error_kw=ekw)
    axis.text(i, m + e+text_gap, '{0:.2f}'.format(m)+'s', ha='center', va='bottom', color=c)


axis.tick_params(axis='both', which='both', length=0)
axis.set_yticks([], [])
axis.set_ylim([0, np.max(measurements)*1.3])
axis.set_xticks(index, simulators, rotation=30, ha="right")
axis.spines[['left', 'right', 'top']].set_visible(False)
axis.set_title('Izhikevich with\nTrace STDP', fontweight='bold')

for xtick, color in zip(axis.get_xticklabels(), colors):
    xtick.set_color(color)



ax[0].text(x=-0.2, y=10, s='A', size=20, weight='bold')
ax[1].text(x=-0.2, y=4.75, s='B', size=20, weight='bold')
#ax[0].text(x=100, y=0, s=' ', size=20, weight='bold')

#ax[0].set_ylabel('compute time')

fig.tight_layout()
#plt.savefig('filename.png', dpi=600)
plt.show()

