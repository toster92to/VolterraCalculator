from scipy.special import comb
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Define the kernel dimension
dim = 20

# Initialize the kernel with zeros
h4 = np.zeros((dim, dim, dim, dim))

# Set values for main diagonal and neighbors
main_diagonal_value = 1.0
decay_factor = 0.2

# Populate the kernel with values
for t1 in range(dim):
    for t2 in range(dim):
        for t3 in range(dim):
            for t4 in range(dim):
                distance = abs(t1 - t2) + abs(t1 - t3) + abs(t1 - t4) + \
                    abs(t2 - t3) + abs(t2 - t4) + abs(t3 - t4)
                if distance <= 5:
                    h4[t1, t2, t3, t4] = (
                        main_diagonal_value - decay_factor * distance) * (-1 if distance % 2 else 1)

# Create meshgrid for the axes
t1, t2, t3 = np.meshgrid(np.arange(dim), np.arange(dim),
                         np.arange(dim), indexing='ij')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.2)

# Initial scatter plot
sc = ax.scatter(t1.flatten(), t2.flatten(), t3.flatten(), c='b', s=50)
ax.set_xlabel('t1')
ax.set_ylabel('t2')
ax.set_zlabel('t3')
ax.set_title('4th Order Volterra Kernel Animation')

# Text annotation for t4 value
t4_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# Slider for manual control of t4
axcolor = 'lightgoldenrodyellow'
ax_t4 = plt.axes([0.25, 0.05, 0.50, 0.03], facecolor=axcolor)
slider_t4 = Slider(ax_t4, 't4', 0, dim-1, valinit=0, valfmt='%0.0f')

# Update function for the plot


def update(frame):
    # Get the slice for the current frame (t4 index)
    current_kernel_slice = h4[:, :, :, frame]

    # Calculate colors and sizes based on the kernel values
    colors = np.where(current_kernel_slice.flatten() > 0, 'g', 'r')
    sizes = np.abs(current_kernel_slice.flatten()) * 100

    # Update the scatter plot
    sc._offsets3d = (t1.flatten(), t2.flatten(), t3.flatten())
    sc.set_array(np.where(current_kernel_slice.flatten() > 0, 1, -1))
    sc.set_sizes(sizes)

    # Update the text annotation
    t4_text.set_text(f"Current t4: {frame}")
    slider_t4.set_val(frame)  # Sync the slider with the animation

    return sc, t4_text


# Animation control
is_paused = False


def onClick(event):
    global is_paused
    if is_paused:
        ani.event_source.start()
    else:
        ani.event_source.stop()
    is_paused = not is_paused


fig.canvas.mpl_connect('button_press_event', onClick)

# Slider update function


def slider_update(val):
    frame = int(slider_t4.val)
    update(frame)
    fig.canvas.draw_idle()


slider_t4.on_changed(slider_update)

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(dim),
                    interval=250, repeat=True)

plt.show()


# ############################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import Slider

# # Define the kernel dimension
# dim = 50

# # Initialize the kernel with zeros
# h4 = np.zeros((dim, dim, dim, dim))

# # Set values for main diagonal and neighbors
# main_diagonal_value = 1.0
# decay_factor = 0.2

# # Populate the kernel with values
# for t1 in range(dim):
#     for t2 in range(dim):
#         for t3 in range(dim):
#             for t4 in range(dim):
#                 distance = abs(t1 - t2) + abs(t1 - t3) + abs(t1 - t4) + \
#                     abs(t2 - t3) + abs(t2 - t4) + abs(t3 - t4)
#                 if distance <= 5:
#                     h4[t1, t2, t3, t4] = (
#                         main_diagonal_value - decay_factor * distance) * (-1 if distance % 2 else 1)

# # Create meshgrid for the axes
# t1, t2, t3 = np.meshgrid(np.arange(dim), np.arange(dim),
#                          np.arange(dim), indexing='ij')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.subplots_adjust(bottom=0.2)

# # Initial scatter plot
# colors = np.where(h4[:, :, :, 0].flatten() > 0, 'g', 'r')
# sizes = np.abs(h4[:, :, :, 0].flatten()) * 100
# sc = ax.scatter(t1.flatten(), t2.flatten(), t3.flatten(), c=colors, s=sizes)
# ax.set_xlabel('t1')
# ax.set_ylabel('t2')
# ax.set_zlabel('t3')
# ax.set_title('4th Order Volterra Kernel Animation')

# # Text annotation for t4 value
# t4_text = ax.text2D(0.05, 0.95, "Current t4: 0", transform=ax.transAxes)

# # Slider for manual control of t4
# axcolor = 'lightgoldenrodyellow'
# ax_t4 = plt.axes([0.25, 0.05, 0.50, 0.03], facecolor=axcolor)
# slider_t4 = Slider(ax_t4, 't4', 0, dim-1, valinit=0, valfmt='%0.0f')


# def update(frame):
#     # Get the slice for the current frame (t4 index)
#     current_kernel_slice = h4[:, :, :, frame]

#     # Calculate colors and sizes based on the kernel values
#     colors = np.where(current_kernel_slice.flatten() > 0, 'g', 'r')
#     sizes = np.abs(current_kernel_slice.flatten()) * 100

#     # Update the scatter plot
#     sc._offsets3d = (t1.flatten(), t2.flatten(), t3.flatten())
#     sc.set_facecolor(colors)
#     sc._sizes = sizes

#     # Update the text annotation
#     t4_text.set_text(f"Current t4: {frame}")
#     fig.canvas.draw_idle()


# def slider_update(val):
#     frame = int(slider_t4.val)
#     update(frame)


# slider_t4.on_changed(slider_update)

# # Create the animation
# ani = FuncAnimation(fig, update, frames=np.arange(dim),
#                     interval=500, repeat=True)

# plt.show()


############################################################################################################


# a Gaussian-like function with a larger variance and exponential decay

def f(x, y):
    return np.exp(-((x - 15)**2 + (y - 15)**2) / 200) * np.exp(-0.01 * (x + y - 30))


x = np.linspace(0, 60, 100)
y = np.linspace(0, 60, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, edgecolor='none')

ax.grid(False)
ax.xaxis.pane.set_edgecolor('k')
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Customize the z-axis
ax.set_zlim(0, 0.6)
ax.zaxis.set_major_locator(plt.LinearLocator(10))
ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.02f}'))

#
cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
cbar.set_label('g(i_1, i_2)')

ax.set_xlabel('i1')
ax.set_ylabel('i2')
ax.set_zlabel('g(i1, i2)')

plt.show()


# Define the function

def f(x, y):
    return np.exp(-((x - 28)**2 + (y - 28)**2) / 200) * np.exp(-0.01 * (x + y - 30))


x = np.linspace(0, 100, 200)
y = np.linspace(0, 100, 200)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.magma, edgecolor='none')

ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Hide the axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')

plt.show()

############################################################################################################
# Define a two-dimensional Gaussian function


def gaussian_2d(x, y, A, mux, muy, sigmax, sigmay):
    return A * np.exp(-((x - mux)**2 / (2 * sigmax**2) + (y - muy)**2 / (2 * sigmay**2)))


# Generate x, y values
x = np.linspace(0, 80, 400)
y = np.linspace(0, 80, 400)
X, Y = np.meshgrid(x, y)

# Parameters for the Gaussian
A = 0.00006  # Amplitude
mux = 15  # Mean along x
muy = 15  # Mean along y
sigmax = 8  # Standard deviation along x
sigmay = 8  # Standard deviation along y

# Calculate Z values
Z = gaussian_2d(X, Y, A, mux, muy, sigmax, sigmay)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.magma)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Adjust label positions closer to the axes
ax.set_xlabel('t1 ', labelpad=-10)
ax.set_ylabel('t2', labelpad=-10)
ax.set_zlabel('h2(t1, t2)', labelpad=-10)
ax.set_title('h2 Kernel')

plt.show()


############################################################################################################
