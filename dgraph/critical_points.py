import numpy as np


class Minimum():
    """
    The Minimum class represents a minimum in the database.
    Parameters
    ----------
    energy : float
    coords : numpy array
        coordinates
    Attributes
    ----------
    energy :
        the energy of the minimum
    coords :
        the coordinates of the minimum.  This is stored as a pickled numpy
        array which SQL interprets as a BLOB.
    fvib :
        the log product of the squared normal mode frequencies.  This is used in
        the free energy calcualations
    pgorder :
        point group order
    invalid :
        a flag that can be used to indicate a problem with the minimum.  E.g. if
        the Hessian has more zero eigenvalues than expected.
    user_data :
        Space to store anything that the user wants.  This is stored in SQL
        as a BLOB, so you can put anything here you want as long as it's serializable.
        Usually a dictionary works best.
    Notes
    -----
    To avoid any double entries of minima and be able to compare them,
    only use `Database.addMinimum()` to create a minimum object.
    See Also
    --------
    Database, TransitionState
    """

    def __init__(self, energy, coords, _id):
        self.energy = energy
        self.coords = np.copy(coords)
        self.invalid = False
        self.id = _id

    def id(self):
        """return the sql id of the object"""
        return self.id

    def __eq__(self, m):
        """m can be integer or Minima object"""
        assert self.id is not None
        if isinstance(m, Minimum):
            assert m.id is not None
            return self.id == m.id
        else:
            return self.id == m

    def __hash__(self):
        _id = self.id
        assert _id is not None
        return _id


class TransitionState():
    """Transition state object
    The TransitionState class represents a saddle point in the database.
    Parameters
    ----------
    energy : float
    coords : numpy array
    min1 : Minimum object
        first minimum
    min2 : Minimum object
        first minimum
    eigenval : float, optional
        lowest (single negative) eigenvalue of the saddle point
    eigenvec : numpy array, optional
        eigenvector which corresponds to the negative eigenvalue
    fvib : float
        log product of squared frequencies for free energy calculation
    pgorder : integer
        point group order
    Attributes
    ----------
    energy :
        The energy of the transition state
    coords :
        The coordinates of the transition state.  This is stored as a pickled numpy
        array which SQL interprets as a BLOB.
    fvib :
        The log product of the squared normal mode frequencies.  This is used in
        the free energy calcualations
    pgorder :
        The point group order
    invalid :
        A flag that is used to indicate a problem with the transition state.  E.g. if
        the Hessian has more than one negaive eigenvalue then it is a higher order saddle.
    user_data :
        Space to store anything that the user wants.  This is stored in SQL
        as a BLOB, so you can put anything here you want as long as it's serializable.
        Usually a dictionary works best.
    minimum1, minimum2 :
        These returns the minima on either side of the transition state
    eigenvec :
        The vector which points along the direction crossing the transition state.
        This is the eigenvector of the lowest non-zero eigenvalue.
    eigenval :
        The eigenvalue corresponding to `eigenvec`.  A.k.a. the curvature
        along the direction given by `eigenvec`
    Notes
    -----
    To avoid any double entries and be able to compare them, only use
    Database.addTransitionState to create a TransitionStateobject.
    programming note: The functions in the database require that
    ts.minimum1._id < ts.minimum2._id.  This will be handled automatically
    by the database, but we must remember not to screw it up
    See Also
    --------
    Database, Minimum
    """

    def __init__(self, energy, coords, min1, min2, eigenval=None, eigenvec=None):
        self.energy = energy
        self.coords = np.copy(coords)
        self.minimum1 = min1  # TODO: Replace with lexographic orderings?
        self.minimum2 = min2
        if eigenvec is not None:
            self.eigenvec = np.copy(eigenvec)
        self.eigenval = eigenval
        self.invalid = False

    def id(self):
        """return the sql id of the object"""
        return 0  # NOTE: Never seems to be used in main tree loops?