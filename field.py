import matplotlib.pyplot as plt

# Set up figure and axis
fig, ax = plt.subplots(figsize=(6, 10))

# Set axis limits and aspect ratio
ax.set_xlim(0, 61)
ax.set_ylim(0, 134)
ax.set_aspect("equal")


# Draw court boundary
plt.plot([0, 61, 61, 0, 0], [0, 0, 134, 134, 0])
plt.plot(
    [(4.2), (61 - 4.2), (61 - 4.2), (4.2), (4.2)],
    [0, 0, 134, 134, 0],
    "k--",
)

# Draw center line and center circle
plt.plot([61 / 2, 61 / 2], [0, 134 / 2 - 19.8], "k--")
plt.plot([61 / 2, 61 / 2], [134 / 2 + 19.8, 134], "k--")

# Short service lines
plt.plot([0, 61], [134 / 2 - 19.8, 134 / 2 - 19.8], "k--")
plt.plot([0, 61], [134 / 2 + 19.8, 134 / 2 + 19.8], "k--")

# Long service lines
plt.plot([0, 61], [7.2, 7.2], "k--")
plt.plot([0, 61], [134 - 7.2, 134 - 7.2], "k--")

# Draw net
plt.plot([0, 61], [134 / 2, 134 / 2], "r-")

# Show plot
plt.show()
