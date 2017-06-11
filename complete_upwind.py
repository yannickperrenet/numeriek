import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags

# makes the plots prettier
plt.style.use('ggplot')


def next_step_approximation(u):
    """ Applying a semi-discrete matrix formulation on a vector u
    to compute a partial derivative of u with respect to t.

    Args:
        u (np.array): vector containing values at time t

    Returns:
        res (np.array): vector with computed values at time (t + 1)
    """
    u_squared = np.array([elt**2 for elt in u])

    # constructs f(u)
    f = W.dot(u_squared)

    # applies r(u)
    f[0] += (1 / (2 * dx)) + v * (1 / (dx**2))

    # Constructs the end result.
    res = v * K.dot(u) + f

    return res


def euler_forward(u_init, dt, t0, t_end):
    """ Applying euler_forward to find a numerical approximation
    starting at t0 and ending at t_end with time-step dt.

    Args:
        u_init (np.array): the initial value at t0
        dt (int): step-size timewize
        t0 (int): starting time
        t_end (int): ending time

    Yields:
        u (np.array): numerial approximation on a certain time
    """
    # Copies the initial condition to u
    u = u_init[:]

    # Computes an approximation for every time-step.
    for t in np.arange(t0, t_end + dt / 3, dt):
        # Only uses a couple of time-steps to create a good looking
        # animation.
        # if not t % (34 * dt):
        yield u

        # Computes the approximation at time t.
        u = u + dt * next_step_approximation(u)

    yield u


def plotting():
    """ Creating and running the simulation. """
    def update_line(u):
        """ Replacing the previous values of u at time t, with the
        new values of u at time (t + 1). """
        # Adds the known value of the Dirichlet boundary.
        u = np.insert(u, 0, 1)

        # Calculates rho with the predetermined values alpha and beta.
        rho = (u - 1) / -2

        # Sets the new values to plot.
        line_u.set_ydata(u)
        line_rho.set_ydata(rho)

        return line_u, line_rho,

    # Creates and initializes the figure.
    fig1 = plt.figure()
    plt.xlim(0, L)
    plt.ylim(-1, 1)

    # Specifies the lines that will be updated and plotted.
    line_u, = plt.plot([], [], 'r-', label='u')
    line_u.set_xdata(street)
    line_rho, = plt.plot([], [], 'b-', label='rho')
    line_rho.set_xdata(street)

    # Creates the animation using the matplotlib.animation module.
    line_ani = animation.FuncAnimation(fig1, update_line,
                                       frames=euler_forward(u, dt, 0, te),
                                       interval=100, blit=True)

    # uncomment the line beneath to save the figure
    # line_ani.save("movie.mp4", metadata={"artist": "goblin_slayer"})

    # shows the animation
    plt.legend()
    plt.title("$\Delta$t = " + str(round(dt, 5)))
    plt.show()


if __name__ == "__main__":
    # initialize parameters
    v, N, L, te = 0.01, 100, 3.0, 5.0

    # determines spatial step-size
    dx = L / (N + 1)

    # sets time-step
    dt = 0.003

    # builds the street
    street = np.arange(0, L + dx / 3, dx)

    # constructs a sparse matrix W to calculate f(u)
    diagonals_w = np.array([N * [-1], (N + 1) * [1], N * [0]])
    W = -(1 / (2 * dx)) * diags(diagonals_w, [-1, 0, 1])

    # constructs sparse matrix K
    diagonals_k = np.array([(N - 1) * [1] + [2], (N + 1) * [-2], N * [1]])
    K = (1 / dx**2) * diags(diagonals_k, [-1, 0, 1])

    # sets up the initial value of the problem
    u = np.array([])
    for x in np.arange(dx, L + dx / 3, dx):
        if x <= L / 3:
            u = np.append(u, 1)
        elif L / 3 <= x <= 2 * L / 3:
            u = np.append(u, 2 - (3 / L) * x)
        else:
            u = np.append(u, 0)

    # creates the animation
    plotting()

    # Determining when the threshold is hit.
    threshold = 0.001
    for i, value in enumerate(euler_forward(u, dt, 0, te)):
        # We want to calculate the threshold for rho, and euler_forward
        # returns a value for u thus we transform.
        value = (value - 1) / -2

        # Applies the Trapezium rule.
        ans = sum(dx * value) - dx * (value[0] + value[-1]) / 2

        if ans < threshold:
            # Print the time at which the threshold is reached.
            print(i * dt)

            break
