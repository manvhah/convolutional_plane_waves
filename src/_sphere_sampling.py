import numpy as np

def fibonacci_sphere(samples=1, randomize=False, semi = False):
    """
    provides uniform grid on a sphere
    """
    import math, random
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    if semi:
        offset *= .5
        increment *= .5

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        if semi:
            points.append([x,z,-y])
        else:
            points.append([x,y,z])

    return np.array(points)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 100

    fib_sphere = fibonacci_sphere(samples=N, randomize=True)
    # fib_sphere[:,1] = -np.abs(fib_sphere[:,1])
    fib_semi   = fibonacci_sphere(samples=N, randomize=True, semi = True)
    unisample_sphere = uniform_sphere(N=N, mode='random')
    unigrid_sphere   = uniform_sphere(N=N, mode='grid')
    tag = ['fibonacci', 'uniform_random', 'uniform_grid']
    dim = ['x','y','z']

    fig = plt.figure(3)
    plt.clf()
    NN, mu = [], []
    for kk, ksphere in enumerate([
                fib_sphere,
                fib_semi,
                unisample_sphere['xyz'],
                unigrid_sphere['xyz']]):

        for ii in range(3):
            ax = plt.subplot(4, 3, 1+kk*3+ii)
            plt.scatter(ksphere[:,ii], ksphere[:,(ii+1)%3],
                    c='#999999',alpha=.15)
            plt.grid()
            plt.xlabel(dim[ii])
            plt.ylabel(dim[(ii+1)%3])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('equal')

        NN.append(ksphere.shape[0])
        GA = np.abs(ksphere.dot(ksphere.T))
        mu.append( np.max(GA-np.diag(np.diag(GA))) )

    s = "{} points on a sphere".format(N)
    plt.text(.1,.95,s,fontsize=12,transform=fig.transFigure, wrap=True)
    s = "fibonacci (N = {}, $\mu$ = {})".format(NN[0], mu[0])
    plt.text(.1,.9,s,fontsize=12,transform=fig.transFigure, wrap=True)
    s = "fibonacci half (N = {}, $\mu$ = {})".format(NN[0], mu[0])
    plt.text(.1,.7,s,fontsize=12,transform=fig.transFigure, wrap=True)
    s = "uniform distribution samples ({},{})".format(NN[1], mu[1])
    plt.text(.1,.5,s,fontsize=12,transform=fig.transFigure, wrap=True)
    s = "grid ({},{})".format(NN[2], mu[2])
    plt.text(.1,.3,s,fontsize=12,transform=fig.transFigure, wrap=True)
    foot()
    plt.show()

