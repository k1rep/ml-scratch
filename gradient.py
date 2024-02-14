import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np

fig = plt.figure(figsize=(8, 8))
ax = axisartist.Subplot(fig, 111)
fig.add_axes(ax)
ax.axis[:].set_visible(False)
ax.axis["x"] = ax.new_floating_axis(0, 0)
ax.axis["x"].set_axisline_style("->", size=1.0)
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["y"].set_axisline_style("-|>", size=1.0)
ax.axis["x"].set_axis_direction("bottom")
ax.axis["y"].set_axis_direction("right")
x = np.linspace(-10, 10, 100)
y = x ** 2 + 3
plt.xlim(-12, 12)
plt.ylim(-10, 100)
plt.plot(x, y)
plt.show()

x, y = np.mgrid[-2:2:20j, -2:2:20j]
z = (x ** 2 + y ** 2)
ax = plt.subplot(111, projection='3d')
ax.set_title('f(x,y)=x^2+y^2')
ax.plot_surface(x, y, z, rstride=9, cstride=1, cmap=plt.cm.Blues_r)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


def grad_1(x):
    return 2 * x


def grad_2(x):
    derivx = 2 * x[0]
    derivy = 2 * x[1]
    return np.array([derivx, derivy])


def grad_descent(x_start, epochs, lr, precision):
    x_current = x_start
    for i in range(epochs):
        grad = grad_1(x_current)
        print('Epoch:', i, 'x:', x_current, 'grad:', grad)
        if abs(grad) < precision:
            break
        x_current -= grad * lr
    print('Local minimum occurs at:', x_current)
    return x_current


def grad_descent_2(x_start, epochs, lr, precision):
    x_current = x_start
    for i in range(epochs):
        grad = grad_2(x_current)
        print('Epoch:', i, 'x:', x_current, 'grad:', grad)
        if np.linalg.norm(x_current, ord=2) < precision:
            break
        x_current = x_current - grad * lr
    print('Local minimum occurs at:', x_current)
    return x_current


if __name__ == '__main__':
    x_start = 5
    epochs = 10000
    lr = 0.1
    precision = 1e-6
    grad_descent(x_start, epochs, lr, precision)
    x_start = np.array([1, -1])
    grad_descent_2(x_start, epochs, lr, precision)
