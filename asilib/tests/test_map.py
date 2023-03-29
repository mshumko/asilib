"""
Test the geographic map creation functions. This does not test the cartopy 
maps.
"""

import matplotlib.testing.decorators


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_create_map_no_subplot'], tol=10, remove_text=True, extensions=['png']
)
def test_create_map_no_subplot():
    """
    Create a geographic map on one subplot.
    """
    import asilib.map
    import matplotlib.pyplot as plt

    ax = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75])
    ax.set_title('Generated via asilib.map.create_map()')
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_create_map_equal_subplots'], tol=10, remove_text=True, extensions=['png']
)
def test_create_map_equal_subplots():
    """
    Create a geographic map on one subplot and random data on the second. The subplots
    have have unequal sizes.
    """
    import asilib.map
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 10))
    bx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], fig_ax=(fig, 211))
    cx = fig.add_subplot(2, 1, 2)
    cx.plot(np.arange(10), np.arange(10))
    fig.suptitle('Two subplots with equal sizes')
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_create_map_unequal_subplots'],
    tol=10,
    remove_text=True,
    extensions=['png'],
)
def test_create_map_unequal_subplots():
    """
    Create a geographic map on one subplot and random data on the second. The subplots
    have have unequal sizes.
    """
    import asilib.map
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 10))
    dx = (3, 1, (1, 2))
    dx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], fig_ax=(fig, dx))
    ex = fig.add_subplot(3, 1, 3)
    ex.plot(np.arange(10), np.arange(10))
    fig.suptitle('Two subplots with unequal sizes')
    return


@matplotlib.testing.decorators.image_comparison(
    baseline_images=['test_create_map_gridspec'], tol=10, remove_text=True, extensions=['png']
)
def test_create_map_gridspec():
    """
    Create a geographic map in a subplot created using gridspec.
    """
    import asilib.map
    import matplotlib.pyplot as plt
    import matplotlib.gridspec

    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(1, 1, fig)
    dx = asilib.map.create_map(lon_bounds=[0, 38], lat_bounds=[50, 75], fig_ax=(fig, gs))
    dx.set_title('Map made using gridspec')
    return
