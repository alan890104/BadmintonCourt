import numpy as np
from typing import List, Tuple


class PerspectiveTransform:
    def __init__(
        self,
        src: List[Tuple[float, float]],
        dst: List[Tuple[float, float]],
    ) -> None:
        self.M = self._get_perspective_projection_matrix(src, dst)

    def _get_perspective_projection_matrix(
        self,
        src: List[Tuple[float, float]],
        dst: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Each input is a 4x2 matrix of (x,y) coordinates of the corners of a rectangle.
        """
        # Convert the input to numpy arrays
        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)

        assert src.shape == dst.shape, "dimension mismatch"
        assert src.shape[0] == 4, "dimension mismatch"

        # Construct the homogeneous system of equations
        A = np.zeros((8, 8))
        b = np.zeros((8, 1))

        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            A[i * 2] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
            A[i * 2 + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
            b[i * 2] = u
            b[i * 2 + 1] = v

        # Solve the system of equations using least squares
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        x = np.append(x, 1)

        # Reshape the solution to a 3x3 matrix
        M = x.reshape((3, 3))

        return M

    def transform(
        self,
        x: List[float],
        y: List[float],
        decimals: int = 8,
    ) -> np.ndarray:
        """
        Applies the perspective transformation defined by the 3x3 matrix self.M to a set of points.

        Parameters
        ----------
        - points (np.ndarray): A Nx2 array of N points in Cartesian coordinates, with shape (N, 2). # 要是左下, 右下, 右上, 左上
        - decimals (int): Number of decimal places to round the transformed points to (default: 8).

        Returns
        -------
        - transformed_points (np.ndarray): A Nx2 array of N transformed points in Cartesian coordinates,
        with shape (N, 2).

        Example:
        >>> src = [[296.8, 658.2], [988.6, 659], [843.6, 274], [438.6, 273.2]]
        >>> dst = [[0, 0], [61, 0], [61, 134], [0, 134]]
        >>> transformer = PerspectiveTransform(src, dst)
        >>> x = [100, 150, 200, 250]
        >>> y = [100, 200, 150, 250]
        >>> transformed_points = transformer.transform(x, y)
        >>> print(transformed_points)
            [[-88.53033602 284.00775591]
            [-54.72805006 183.77898745]
            [-55.11171215 228.12637239]
            [-30.93211887 148.15418828]]
        """
        assert len(x) == len(y), "dimension mismatch"

        # Convert the input to a numpy array
        points = np.array([x, y]).T
        points = np.array(points, dtype=np.float32)

        # Check that the input has the correct shape
        assert points.shape[1] == 2, "dimension mismatch"

        # Add a column of ones to the points to turn them into homogeneous coordinates
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))

        # Apply the perspective transformation to the points
        transformed_points_homogeneous = np.dot(self.M, points_homogeneous.T)

        # Convert the transformed points back to Cartesian coordinates
        transformed_points = (
            transformed_points_homogeneous[:2] / transformed_points_homogeneous[2]
        ).T
        transformed_points[abs(transformed_points) < 10**-decimals] = 0.0
        return np.around(transformed_points, decimals)


if __name__ == "__main__":
    src = [[296.8, 658.2], [988.6, 659], [843.6, 274], [438.6, 273.2]]
    dst = [[0, 0], [61, 0], [61, 134], [0, 134]]
    transformer = PerspectiveTransform(src, dst)

    x = [100, 150, 200, 250]
    y = [100, 200, 150, 250]
    transformed_points = transformer.transform(x, y)
    print(transformed_points)
