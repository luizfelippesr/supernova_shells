import numpy as np
import numba as nb

class ShellModel:
    """
    A model for the evolution of a supernova shell

    Parameters
    ----------
    V0 : float
        Peak velocity of the gas (pc/yr)
    R : float
        Radius of the peak velocity (pc)
    a, b : float
        Parmeters controling the shape of the velocity distribution
    elapsed_time : float
        Elapsed time in years
    """
    def __init__(self, V0=0.0153, a=1.3, b=10, R=50, elapsed_time=1300):
        assert a != 1
        self.a = a
        assert b != 0
        self.b = b
        self.V0 = V0
        self.R = R
        self.t = elapsed_time

    def final_radius(self, x):
        """
        Computes Final (Eulerian) radius from Initial (Lagrangian)
        """
        return _final_radius(x, self.t, self.a, self.b, self.R, self.V0)

    def initial_radius(self, x):
        """
        Computes Initial (Lagrangian) radius from Final (Eulerian)
        """
        return _initial_radius(x, self.t, self.a, self.b, self.R, self.V0)

    def dr_dr0(self, r, r0):
        """
        Derivative dr_dr0
        """
        return _dr_dr0(r, r0, self.a, self.b, self.R)

# The following decorators vectorize (i.e. allow a scalar funtion to be
# applied on numpy arrays) and speed up (using just-in-time-compilation)
# the calculations in the decorated functions
vect6 = nb.vectorize(['float64(float64, float64, float64, float64, float64, float64)'])
vect5 = nb.vectorize(['float64(float64, float64, float64, float64, float64)'])

@vect5
def _dr_dr0(r, r0, a, b, R):
    """
    Derivative dr_dr0
    """
    # If we start inside the shell
    if (r0 < R):
        # If we finish inside the shell
        if (r<=R):
            value = (r/r0)**a;
        # If we finish outside the shell
        else:
            value = np.exp(b*(1-r/R))*(R/r0)**a
    # If we start outside
    else:
        value = np.exp(b*(r0-r)/R)
    return value

@vect6
def _final_radius(r0, t, a, b, R, V0):
    """
    Computes Final (Eulerian) radius from Initial (Lagrangian)


    This function returns you r from initial radius, r0,
    time, t, and given parameters of the field: a, b, R, V0
    a != 1 and b != 0
    """
    # If we start inside the shell
    if (r0 < R):
        # Time it takes to reach from r0 to the R is Tau
        Tau = R/(1-a)/V0 * ( 1 - (r0/R)**(1-a) );

        # if Tau is more than t, then it means that it takes more time to reach
        # parameter radius R than time to reach final position r
        # In other words, r<R
        if (Tau >= t):
            r = ( r0**(1-a) + (1-a)*V0*t/R**a )**(1/(1-a));

        # if Tau is less than t, then it means that it takes less time to reach
        # R than r
        # In other words, r>=R
        # Here, we start within R, but finish otside
        else:
            r = R * ( 1 + np.log( 1 + b*V0*t/R - b*(1 - (r0/R)**(1-a))/(1-a) )/b );

    # If we start outside the shell
    else:
        r = R * ( 1 + np.log(np.exp(b*(r0/R-1)) + b/R*V0*t)/b )
    return r

@vect6
def _initial_radius(r, t, a, b, R, V0):
    # Initial (Lagrangian) radius from Final (Eulerian)
    #   Look at finFromInit function

    # If we finish inside the shell
    if (r<=R):
        r0 = ( r**(1-a) - (1-a)*V0*t/R**a )**(1/(1-a))
    # If we finish outside the shell
    else:
        # Time it takes to go from R to r is Tau
        Tau = R*( np.exp(b*(r/R-1)) - 1 )/b/V0

        # If Tau is bigger than t, then we started from outside of the shell
        # In other words r0 >= R
        if (Tau >= t):
            r0 = R * ( 1 + np.log(np.exp(b*(r/R-1)) - b/R*V0*t)/b )

        # If Tau is smaller than t, then we started from inside the shell
        # and then moved outside
        # In other words r0 < R
        else:
            r0 = R*( 1 + (1-a)*( np.exp(b*(r/R-1)) - 1 - b*V0*t/R )/b )**(1/(1-a))

    return r0
