import matplotlib.pyplot as plt
import numpy as np
import csv

def load2(filename):
    sim = []
    col = []
    num = []
    mes = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i>0:
                sim.append(lookup[row[1]][0])
                col.append(lookup[row[1]][1])
                num.append(int(row[2]))
                mes.append(float(row[3]))
    return np.array(sim), np.array(col), np.array(num), np.array(mes)

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
          'pymonntorch_fast_LIF_cpu.py': ['PymoNNtorch CPU ', color1],
          'pymonntorch_fast_LIF_cuda.py': ['PymoNNtorch GPU', color1],
          'pymonntorch_slow_LIF_cpu.py': ['PymoNNtorch CPU', color1],# naive
          'pymonntorch_slow_LIF_cuda.py': ['PymoNNtorch GPU', color1],# naive

          'brian_izh.py': ['Brian2', color2],
          'brian_izh_cpp.py': ['Brian2 C++', color2],
          'brian_izh_cuda.py': ['Brian2 GPU', color2],
          'pymonnto_izh.py': ['PymoNNto', color11],
          'pymonntorch_izh_cpu.py': ['PymoNNtorch CPU', color1],
          'pymonntorch_izh_cuda.py': ['PymoNNtorch GPU', color1],
          'pynn_nest_izh.py': ['Nest (PyNN)', color3]
          }

markers_lookup = {"⋅⋅⋅ Pymonntorch GPU": ':',
"⋅⋅⋅ PymoNNtorch GPU": ':',
		  "-- Pymonntorch CPU": '--',
		  "-- Pymonntorch CPU ": '--',
		  "-- PymoNNtorch CPU": '--',
		  "-- PymoNNtorch CPU ": '--',
		  "— PymoNNto": '-',
"— Nest": '-',
"⋅⋅⋅ Brian2 GPU": ":",
"-- Brian2 C++": "--",
"— Brian2": "-",		
'— Nest (PyNN)': "-",  
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
fig.set_figwidth(5)#12
fig.set_figheight(4)

axis = ax
sim_col, data = load('Results/Swift-SF315-51G/Simple2.csv')

avg_measurements = np.mean(data, axis=0)
avg_speed_ups = np.max(avg_measurements)/avg_measurements
speed_up_err = np.std(np.max(avg_measurements)/data, axis=0)

text_gap = np.max(avg_speed_ups)*0.01

simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
index = list(range(len(sim_col)))
index = [6,5,4,2,1,3]#5,6,7, 2,3,1


for i, s, m, e, c in zip(index, simulators, avg_speed_ups, speed_up_err, colors):
    #axis.plot([i-0.4,i+0.4],[m*14,m*14], color='red',)
    # axis.barh(i, m, width=0.8, color=c, yerr=e, error_kw=ekw)
    axis.barh(i, m , color=c, xerr=e, error_kw=ekw)
    axis.text(m+e+text_gap+20, i-0.02, '{0:.1f}'.format(m)+'x', ha='left', va='center', color=c)  # , color='gray'

axis.tick_params(axis='both', which='both', length=0)
axis.set_xticks([], [])
#axis.set_ylim([0, np.max(measurements)*1.5])#1.3
# axis.set_xticks(np.array(index), simulators, rotation=30, ha="right")
axis.set_yticks(np.array(index), simulators)
# axis.spines[['left', 'right', 'top']].set_visible(False)
axis.spines[['bottom', 'right', 'top']].set_visible(False)
axis.set_title('Simple LIF with\nOne Step STDP', size=10, fontweight='bold') #Simple LIF with\nOne Step STDP\n optimized vs naive

for ytick, color in zip(axis.get_yticklabels(), colors):
    ytick.set_color(color)

axis.hlines(3.5, 0, 1000, linestyles='--', color='black', linewidth=1)

axis.text(1000, 3.55, 'optimized', ha='right', va='bottom', color=(0,0,0,1))
axis.text(1000, 3.15, 'naive', ha='right', va='bottom', color=(0,0,0,1))

fig.tight_layout()
plt.savefig('new_measurements_x2.png', dpi=600)
plt.show()


lookup = {'brian_LIF.py': [ '— Brian2', color2],
          'brian_LIF_cpp.py': ['-- Brian2 C++', color2],
          'brian_LIF_gpu.py': ['⋅⋅⋅ Brian2 GPU', color2],
          'nest_native_LIF.py': ['— Nest', color3],
          'pymonnto_fast_LIF.py': ['— PymoNNto', color11],
          'pymonnto_slow_LIF.py': ['— PymoNNto', color11],# naive
          'pymonntorch_fast_LIF_cpu.py': ['-- PymoNNtorch CPU ', color1],
          'pymonntorch_fast_LIF_cuda.py': ['⋅⋅⋅ PymoNNtorch GPU', color1],
          'pymonntorch_slow_LIF_cpu.py': ['-- PymoNNtorch CPU', color1],# naive
          'pymonntorch_slow_LIF_cuda.py': ['⋅⋅⋅ PymoNNtorch GPU', color1],# naive

          'brian_izh.py': ['— Brian2', color2],
          'brian_izh_cpp.py': ['-- Brian2 C++', color2],
          'brian_izh_cuda.py': ['⋅⋅⋅ Brian2 GPU', color2],
          'pymonnto_izh.py': ['— PymoNNto', color11],
          'pymonntorch_izh_cpu.py': ['-- PymoNNtorch CPU', color1],
          'pymonntorch_izh_cuda.py': ['⋅⋅⋅ PymoNNtorch GPU', color1],
          'pynn_nest_izh.py': ['— Nest (PyNN)', color3]
          }


#1
###############################################################################
#2

fig, ax = plt.subplots(1, 2, width_ratios=[55, 45])
fig.set_figwidth(8)#12
fig.set_figheight(4)



fig.suptitle('Izhikevich with\nTrace STDP', fontweight='bold')


axis = ax[0]
axis.axvline(2500, c='lightgray', linestyle='--')
sim, col, num, mes = load2('Results/Swift-SF315-51G/IZH.csv')
for s in np.flip(np.unique(sim)):
    idx = sim==s
    x = np.sort(np.unique(num[idx]))
    y = np.array([np.mean(mes[idx][num[idx]==n]) for n in x])
    e = np.array([np.std(mes[idx][num[idx]==n]) for n in x])
    axis.fill_between(x, y - e, y + e, alpha=0.5, edgecolor=col[idx][0], facecolor=col[idx][0], linewidth=0) #, c=col[idx][0]
    # axis.plot(x, y, c=col[idx][0])
    axis.plot(x, y, c=col[idx][0], linestyle=markers_lookup[s], label=s)
axis.semilogy()
axis.tick_params(axis='x', which='both', length=3)
axis.tick_params(axis='y', which='both', length=0)
axis.set_yticks([], [])
axis.set_ylabel('Simulation Time (log scale)')

axis.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000], [0, 2500, 5000, 7500, 10000, 12500, 15000])
axis.spines[['left', 'right', 'top']].set_visible(False)
axis.set_xlim([0, 15000])
axis.set_title('Number of Neurons', x=0.85, y=0, pad=-14, fontsize=10)
# fig.legend(handles, labels, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0.01), prop={'size':8} )





axis = ax[1]
sim_col, data = load('Results/Swift-SF315-51G/Izhikevich.csv')

avg_measurements = np.mean(data, axis=0)
avg_speed_ups = np.max(avg_measurements)/avg_measurements
speed_up_err = np.std(np.max(avg_measurements)/data, axis=0)

text_gap = np.max(avg_speed_ups)*0.01

simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
#index = list(range(len(sim_col)))
index = [3,4,2,5,6,7,1]

for i, s, m, e, c in zip(index, simulators, avg_speed_ups, speed_up_err, colors):
    axis.barh(i, m, color=c, xerr=e, error_kw=ekw)
    axis.text(m + e+text_gap + 0.2, i-0.02, '{0:.1f}'.format(m)+'x', ha='left', va='center', color=c)


axis.tick_params(axis='both', which='both', length=0)
axis.set_xticks([], [])
axis.set_xlim([0, np.max(avg_speed_ups)*1.3])
axis.set_yticks(index, simulators, ha="right")
axis.spines[['bottom', 'right', 'top']].set_visible(False)

for ytick, color in zip(axis.get_yticklabels(), colors):
    ytick.set_color(color)



# ax[0].text(x=0, y=160, s='C', size=17)
# ax[1].text(x=0, y=7.5, s='D', size=17)
ax[0].set_title("C", loc='left')
ax[1].set_title("D", loc='left')

#ax[0].text(x=100, y=0, s=' ', size=20, weight='bold')

#ax[0].set_ylabel('compute time')

fig.tight_layout()
plt.savefig('new_measurements_x.png', dpi=600)
plt.show()


#2
###############################################################################
#3

            #print(i, row)
            #if i==0:
            #    sim_col = [lookup[s] for s in row[1:]]
            #else:
            #    measurements.append([float(s) for s in row])

    #return sim_col, np.array(measurements)[:,1:] #remove enumeration in first column

#2
###############################################################################
#3

fig, ax = plt.subplots(1, 2, width_ratios=[55, 45])
fig.set_figwidth(8)#12
fig.set_figheight(4)

fig.suptitle('Simple LIF with\nOne Step STDP', fontweight='bold')
axis = ax[0]
axis.axvline(7500, c='lightgray', linestyle='--')
sim, col, num, mes = load2('Results/Swift-SF315-51G/LIF.csv')
for s in np.flip(np.unique(sim)):
    idx = sim==s
    x = np.sort(np.unique(num[idx]))
    y = np.array([np.mean(mes[idx][num[idx]==n]) for n in x])
    e = np.array([np.std(mes[idx][num[idx]==n]) for n in x])
    axis.fill_between(x, y - e, y + e, alpha=0.5, edgecolor=col[idx][0], facecolor=col[idx][0], linewidth=0) #, c=col[idx][0]
    # axis.plot(x, y, c=col[idx][0])
    axis.plot(x, y, c=col[idx][0], linestyle=markers_lookup[s], label=s)
handles, labels = axis.get_legend_handles_labels()
axis.semilogy()
axis.tick_params(axis='x', which='both', length=3)
axis.tick_params(axis='y', which='both', length=0)
axis.set_yticks([], [])
axis.set_ylabel('Simulation Time (log scale)')
axis.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000], [0, 2500, 5000, 7500, 7500, 12500, 15000])
axis.set_title('Number of Neurons', x=0.85, y=0, pad=-14, fontsize=10)
axis.spines[['left', 'right', 'top']].set_visible(False)
axis.set_xlim([0, 15000])


axis = ax[1]
sim_col, data = load('Results/Swift-SF315-51G/Simple.csv')

avg_measurements = np.mean(data, axis=0)
avg_speed_ups = np.max(avg_measurements)/avg_measurements
speed_up_err = np.std(np.max(avg_measurements)/data, axis=0)

text_gap = np.max(avg_speed_ups)*0.01

#measurements = np.mean(data, axis=0)
#err = np.std(data, axis=0)
simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
#index = list(range(len(sim_col)))
index = [3,4,2,1,5,6,7]

for i, s, m, e, c in zip(index, simulators, avg_speed_ups, speed_up_err, colors):
    axis.barh(i, m, color=c, xerr=e, error_kw=ekw)
    axis.text(m + e+text_gap+0.2, i-0.02, '{0:.1f}'.format(m)+'x', ha='left', va='center', color=c)



axis.tick_params(axis='both', which='both', length=0)
axis.set_xticks([], [])
axis.set_xlim([0, np.max(avg_speed_ups)*1.3])
axis.set_yticks(index, simulators, ha="right")
axis.spines[['bottom', 'right', 'top']].set_visible(False)

for ytick, color in zip(axis.get_yticklabels(), colors):
    ytick.set_color(color)



ax[0].set_title("A", loc='left')
ax[1].set_title("B", loc='left')

fig.tight_layout()
# plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=0.1)

# ax[0].text(x=0, y=15, s='A', size=17)
# ax[1].text(x=0, y=7.5, s='B', size=17)

plt.savefig('new_measurements_n.png', dpi=600)
plt.show()


