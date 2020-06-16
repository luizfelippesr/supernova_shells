import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
def get_latex_units(q):
    return q.unit._repr_latex_().replace('$','')

def plot_scalar_xy(grid, scalar_field, name='n', colormesh=True,
                   pos=None, ax=None, fig=None, **kwargs):
    """
    Plots a slice of a scalar field defined on a 3D cartesian grid
    """
    if ax is None:
        if fig is None:
            fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
        
    if pos is None:
        # If no position is supplied, take the middle!
        pos = grid.resolution[2]//2
        
    if len(scalar_field.shape)==3:
        im_slice = scalar_field[:,:,pos]
    else:
        im_slice = scalar_field
    
    if 'cmap' not in kwargs:
        if scalar_field.unit.is_equivalent(u.rad):
            kwargs['cmap'] = 'twilight_shifted'
    
    if colormesh:
        im = ax.pcolormesh(grid.x[:,:,pos].value, grid.y[:,:,pos].value, 
                           im_slice.value, **kwargs)
    else:
        im = ax.contourf(grid.x[:,:,pos], grid.y[:,:,pos], 
                         im_slice.value, **kwargs)
    
    ax.set_aspect(1)
    ax.set_xlabel(r'$x\;[\rm pc]$')
    ax.set_ylabel(r'$y\;[\rm pc]$')
    cax = plt.colorbar(im, ax=ax)
    cax.set_label(r'${}\;\left[\,{}\,\right]$'.format(name,
                                  get_latex_units(scalar_field)))
    return fig
    
def plot_vector_xy(grid, vector_field, skip=3, name=r'\mathbf{B}',
                   pos=None, ax=None, fig=None, show_z_component=False,
                   quiver_color='orange', contour_alpha=1,**kwargs):
    """
    Plots a slice of a vector field defined on a 3D cartesian grid
    """
    if ax is None:
        if fig is None:
            fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111)
    
    if pos is None:
        # If no position is supplied, take the middle!
        pos = grid.resolution[2]//2
        
    Bx, By, Bz = vector_field
    if show_z_component:
        B = Bz
        vmax = np.max(np.abs(Bz.value))
        if ('vmin' not in kwargs) and ('vmax' not in kwargs):
            kwargs['vmin'] = -vmax
            kwargs['vmax'] = vmax
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'coolwarm'
    else:
        B = np.sqrt(Bx**2+By**2+Bz**2)
        name = '|'+name+'|'
    
    im = ax.contourf(grid.x[:,:,pos],grid.y[:,:,pos], B[:,:,pos], 
                     alpha=contour_alpha, **kwargs)
    # Quiver does not handle units well. Does, we select the values instead
    ax.quiver(grid.x[::skip,::skip,pos].value, grid.y[::skip,::skip,pos].value, 
           Bx[::skip,::skip,pos].value, By[::skip,::skip,pos].value, color=quiver_color)

    ax.set_aspect(1)
    ax.set_xlabel(r'$x\;[\rm pc]$')
    ax.set_ylabel(r'$y\;[\rm pc]$')
    cax = plt.colorbar(im, ax=ax)
    cax.set_label(r'${}\;\left[\,{}\,\right]$'.format(name, get_latex_units(B)))
    return fig