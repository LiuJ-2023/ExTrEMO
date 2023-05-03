import numpy as np

def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# Test
if __name__ == "__main__":

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    ########### 2D example ######################
    # a = np.random.rand(20,2)
    # is_efficient = is_pareto_efficient_simple(a)
    #
    # plt.plot(a[:,0], a[:,1],'bo')
    # for i in range(20):
    #     if is_efficient[i]:
    #         plt.plot(a[i,0],a[i,1],'ro')
    #
    # plt.show()

    ########### 3D example ######################
    a = np.random.rand(200, 3)
    is_efficient = is_pareto_efficient_simple(a)

    x = a[is_efficient,0]
    y = a[is_efficient,1]
    z = a[is_efficient,2]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    ax.scatter3D(x, y, z, color="blue")

    plt.show()