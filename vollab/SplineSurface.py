"""
    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    A two dimensional surface composed of cubic splines,
     parallel with the x-axis at fixed y co-ordinates.

"""
import numpy as np

from scipy.interpolate import CubicSpline


class SplineSurface(object):
    """
       Two dimensional surface composed of cubic splines parallel with x-axis.
    """

    @staticmethod
    def _key(variable):
        """

        Return something that can be used as a key in a dictionary.
            Definitely do not use floating point types as keys in dictionaries!

        Args:
            variable: The variable.

        Returns:
            The key.

        """
        return str(variable)

    def __init__(self, x_axis, y_axis, surface_matrix, y_fine_axis=None):
        """

        Args:
            x_axis: The x-axis.
            y_axis: The y-axis.
            surface_matrix: The surface matrix.
            y_fine_axis: Co-ordinates to fill in the surface using cross interpolation
        """
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.surface_matrix = surface_matrix

        self.y_lookup = dict()
        for y_coord, values in zip(y_axis, surface_matrix):
            self.y_lookup[self._key(y_coord)] = CubicSpline(x_axis, values)

        if len(y_fine_axis) > 0:
            self.x_lookup = dict()
            values_by_y = np.empty([len(y_axis)])
            for x_coord in x_axis:
                for i, y_coord in enumerate(y_axis):
                    values_by_y[i] = self.y_lookup[self._key(y_coord)](x_coord)
                self.x_lookup[self._key(x_coord)] = CubicSpline(y_axis, values_by_y)

            values_by_x = np.empty([len(x_axis)])
            for y_coord in y_fine_axis:
                if self._key(y_coord) not in self.y_lookup:
                    for i, x_coord in enumerate(x_axis):
                        values_by_x[i] = self.x_lookup[self._key(x_coord)](y_coord)
                    self.y_lookup[self._key(y_coord)] = CubicSpline(x_axis, values_by_x)

    def __call__(self, x_coord, y_coord):
        """

        Return the value for co-ordinates x_coord and y.

        Args:
            x_coord: The x_coord co-ordinate.
            y_coord: The y co-ordinate.

        Returns:
            The value.
        """
        return self.y_lookup[self._key(y_coord)](x_coord)

    def _axis(self, y_coord):
        """
            Return the axis of the spline for co-ordinate y.
        Args:
            y_coord: The y co-ordinate.

        Returns:
            Numpy array of the axis values.

        """
        return np.array([x for x in self.y_lookup[self._key(y_coord)].x], dtype=np.float32)

    def values(self, y_coord):
        """
            Return the surface values along the x-axis for the y co-ordinate.
        Args:
            y_coord: The y co-ordinate.

        Returns:

        """
        return np.array([self.__call__(x, y_coord) for x in self._axis(y_coord)], dtype=np.float32)


