import matplotlib.pyplot as plt
import numpy as np

fig = plt.Figure(figsize=(9, 9))
ax = plt.subplot()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

metrics = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"]

min_lm_to_scores = {
    1: [39.051, 35.89, 44.575, 42.664, 46.014, 50.213, 57.883, 65.977],
    3: [39.431, 36.208, 45.013, 42.654, 47.013, 51.162, 58.509, 66.289],
    5: [40.101, 36.321, 46.472, 42.333, 47.785, 51.805, 60.11, 66.379],
    7: [39.911, 36.251, 46.001, 41.841, 48.39, 51.498, 59.979, 66.427],
    9: [40.183, 36.055, 46.73, 41.115, 49.179, 52.285, 60.095, 66.606],
    11: [40.477, 35.849, 47.784, 40.681, 49.451, 52.976, 60.866, 66.65],
    13: [40.693, 35.808, 48.226, 40.936, 49.043, 53.846, 59.898, 66.644],
    15: [39.679, 35.458, 46.383, 39.93, 50.063, 51.673, 60.911, 66.71],
    17: [39.651, 35.175, 46.71, 39.473, 50.311, 52.387, 60.598, 66.756],
    19: [38.487, 33.949, 45.498, 38.015, 50.051, 50.94, 60.02, 66.659],
    21: [38.19, 33.445, 45.387, 37.368, 50.162, 50.364, 60.387, 66.572],
}
keep_frames_to_scores = {
    1: [39.108, 35.809, 44.767, 39.791, 51.199, 49.161, 61.841, 66.881],
    3: [39.63, 36.035, 45.617, 40.375, 50.591, 50.375, 61.402, 66.803],
    5: [39.765, 36.038, 45.94, 40.627, 50.06, 50.681, 61.271, 66.757],
    7: [40.478, 36.046, 47.579, 40.876, 49.652, 53.008, 60.621, 66.689],
    9: [40.477, 35.849, 47.784, 40.681, 49.451, 52.976, 60.866, 66.65],
    11: [40.693, 35.808, 48.226, 40.936, 49.043, 53.846, 59.898, 66.644],
    13: [39.43, 35.687, 45.293, 40.896, 48.799, 51.918, 57.982, 66.635],
    15: [38.23, 34.212, 44.347, 40.747, 48.663, 52.419, 58.255, 66.581],
}
linestyles = ["solid", "dashed", "dotted"]

for i, metric in enumerate(metrics):
    ax.plot(
        min_lm_to_scores.keys(),
        [val[i] for val in min_lm_to_scores.values()],
        label=metric,
        linestyle=linestyles[i],
    )
    if i == 2:
        break
ax.set_xlabel("# min. landmarks")
ax.set_xticks(list(min_lm_to_scores.keys()))
ax.vlines(13, 0, 48.5, color="k", linewidth=1, linestyle="dotted")
ax.set_ylim(32.5, 50)
plt.legend()

plt.show()

ax = plt.subplot()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


for i, metric in enumerate(metrics):
    ax.plot(
        keep_frames_to_scores.keys(),
        [val[i] for val in keep_frames_to_scores.values()],
        label=metric,
        linestyle=linestyles[i],
    )
    if i == 2:
        break
ax.set_xlabel("# max. frames after lost")
ax.set_xticks(list(keep_frames_to_scores.keys()))
ax.vlines(11, 0, 48.5, color="k", linewidth=1, linestyle="dotted")
ax.set_ylim(32.5, 50)
plt.legend()

plt.show()
