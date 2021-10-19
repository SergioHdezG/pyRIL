import numpy as np
import matplotlib.pyplot as plt

colors = ["tab:blue", "tab:green", "tab:red", "tab:orange", "tab:purple"]

######## Exp 1 road ################
algos = ["Expert", "BC", "RL", "DeepIRL", "GAIL"]
disp = [465.27, 224.81, 363.19, 449.42, 461.16]
path = [663.25, 316.29, 488.77, 567.16, 606.12]
speed = [25.17, 23.92, 28.64, 30.28, 30.34]
off = [0., 1., 0.4, 3.95, 1.6]
coll = [0., 1., 0.4, 0., 0.]


######### Exp 1 street ################
algos = ["Expert", "BC", "RL", "DeepIRL", "GAIL"]
disp = [188.09, 183.50, 186.50, 180.54, 190.00]
path = [305.25,  294.85, 278.35, 255.5, 279.41]
speed = [19.68, 19.06, 23.80, 21.31, 25.37]
off = [0., 0., 0.25, 0.6, 3.4]
coll = [0., 0., 0.25, 0., 0.15]


# ######### Exp 2 road ################
# colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
# algos = ["Expert", "BC", "DeepIRL", "GAIL"]
# disp = [113.12, 104.69, 197.49, 9.93]
# path = [176.16,  151.41, 294.72, 15.4]
# speed = [14.91, 15.8, 18.83, 2.9]
# off = [0., 0., 3.2, 0.2]
# coll = [0., 0., 0.2, 0.2]

X = speed
Y = disp
scale = (np.array(coll) + 0.1) * 100
etiquetas = algos


fig, ax = plt.subplots(figsize=(6,4))
plt.xlabel("speed (km/h)")
plt.ylabel("distance (meters)")
plt.title("Scenario 2 (street)")
ax.set_xlim([np.min(X) - np.min(X)*0.05, np.max(X)+ (np.max(X)*0.05)])
ax.set_ylim([np.min(Y) - np.min(Y)*0.05, np.max(Y)+ (np.max(Y)*0.05)])
scatter = ax.scatter(X, Y, s=scale, color=colors)
for i, label in enumerate(etiquetas):

    ax.annotate(label, (X[i], Y[i]), xytext=(4, 0), textcoords='offset points')
    # plt.annotate(label, (X[i], Y[i]))

# produce a legend with the unique colors from the scatter
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Classes")
# ax.add_artist(legend1)

# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)

lab = []
for l in labels:
    l = l.split("{")
    l = float(l[-1][:-2])
    l = l/100. - 0.1
    l = '{:.1f}'.format(l)
    lab.append('$\\mathdefault{' + l + '}$')
labels = lab
legend2 = ax.legend(handles, labels, loc="lower right", title="Collision/episode")

plt.show()
