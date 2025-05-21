import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

import matplotlib.pyplot as plt

print("Starting plot...")

plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Debugger Test")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)

plt.show()

print("Plot should be visible.")
