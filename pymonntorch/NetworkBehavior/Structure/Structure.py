"""
This module includes functions and classes to handle structured NeuronGroups.
"""
import torch
from pymonntorch.NetworkCore.Behavior import Behavior


def vec_to_mat(vec, repeat_count):
    """
    Stack `repeat_count` numbers of the input vector on first dimension.

    Args:
        vec (torch.tensor): The vector to replicate.
        repeat_count (int): Number of replicas.

    Returns:
        torch.tensor: A matrix of the stacked `vec`s for `repeat_count` times.
    """
    return torch.stack([vec] * repeat_count, dim=0)


def vec_to_mat_transposed(vec, repeat_count):
    """
    Stack `repeat_count` numbers of the input vector on second dimension.

    Args:
        vec (torch.tensor): The vector to replicate.
        repeat_count (int): Number of replicas.

    Returns:
        torch.tensor: A matrix of the stacked `vec`s for `repeat_count` times.
    """
    return torch.stack([vec] * repeat_count, dim=1)


def rotation_matrix(axis, theta, dtype=torch.float):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given `axis` by `theta` radians.

    Args:
        axis (int): The axis.
        theta (float): The angle of rotation in radians.
        dtype (type): Type of the tensor values. The default is `torch.float`.

    Returns:
        torch.tensor: The rotation matrix.
    """
    axis = torch.tensor(axis, dtype=dtype)
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def get_squared_dim(n_neurons, depth=1):
    """
    Get the squared dimension of a square matrix that can hold `n_neurons`
    neurons, with `depth` layers. The neurons are structured in a square-like 
    manner.

    Args:
        n_neurons (int): The desired number of neurons in the `NeuronGroup`.
        depth (int): The depth of the structured `NeuronGroup`. The default is 1.

    Returns:
        NeuronDimension: An instance of `NeuronDimension` behavior with the given depth \
            and calculated height and depth.
    """
    divider = int(torch.trunc(torch.sqrt(torch.tensor(n_neurons))))

    while divider > 1 and n_neurons % divider != 0:
        divider -= 1

    width = divider
    height = torch.div(n_neurons, divider, rounding_mode="floor") if divider > 0 else 0

    print(f"Set population size to {depth}x{height}x{width}.")

    return NeuronDimension(width=width, height=height, depth=depth)


class NeuronDimension(Behavior):
    """
    The Behavior that defines structure for a `NeuronGroup`.
    It overrides the `size` variable of the `NeuronGroup` and adds `x`, `y`, and `z` vectors,
    as well as `width`, `height` and `depth` variables. The Behaviour is special, because its
    `initialize` function is executed when the `NeuronGroup` is created rather than
    when network.initialize() is called. The neurons are arranged in a 3-dimensional grid
    with size=width * height * depth. Because is overrides the `size` variable, it does not
    have to be added in the behaviour dictionary directly, but also indirectly
    ( `NeuronGroup(size=NeuronDimension(),...)` ). In this case it will be added to
    position 0 in the dictionary.

    Args:
        width (int): Width of the `NeuronGroup`. The default is 1.
        height (int): Height of the `NeuronGroup`. The default is 1.
        depth (int): Depth of the `NeuronGroup`. The default is 1.
    """

    initialize_on_init = True

    def __init__(
        self, *args, depth=1, height=1, width=1, input_patterns=None, **kwargs
    ):
        super().__init__(
            *args,
            depth=depth,
            height=height,
            width=width,
            input_patterns=input_patterns,
            **kwargs,
        )

    def set_position(self, width, height, depth):
        """Set the coordinate of neurons by setting vectors `x`, `y`, and `z`.

        Args:
            width (int): Width of the neurons.
            height (int): Height of the neurons.
            depth (int): Depth of the neurons.
        """
        self.neurons.x = (
            torch.arange(
                0,
                self.neurons.size,
                dtype=self.neurons.def_dtype,
                device=self.neurons.device,
            )
            % width
        )
        self.neurons.y = (
            torch.div(
                torch.arange(
                    0,
                    self.neurons.size,
                    dtype=self.neurons.def_dtype,
                    device=self.neurons.device,
                ),
                width,
                rounding_mode="floor",
            )
            % height
        )
        self.neurons.z = (
            torch.div(
                torch.arange(
                    0,
                    self.neurons.size,
                    dtype=self.neurons.def_dtype,
                    device=self.neurons.device,
                ),
                (width * height),
                rounding_mode="floor",
            )
            % depth
        )

    def get_area_mask(self, xmin=0, xmax=-1, ymin=0, ymax=-1, zmin=0, zmax=-1):
        """Returns a mask tensor with the same shape as the NeuronGroup
        with the given start and end points.

        Args:
            xmin (int): Start point in x axis. The default is 0.
            xmax (int): End point in x axis. The default is -1.
            ymin (int): Start point in y axis. The default is 0.
            ymax (int): End point in y axis. The default is -1.
            zmin (int): Start point in z axis. The default is 0.
            zmax (int): End point in z axis. The default is -1.

        Returns:
            torch.BoolTensor: The mask tensor.
        """
        if xmax < 1:
            xmax = self.width - xmax
        if ymax < 1:
            ymax = self.height - ymax
        if zmax < 1:
            zmax = self.depth - zmax

        result = torch.zeros(
            self.depth,
            self.height,
            self.width,
            dtype=torch.bool,
            device=self.neurons.device,
        )
        result[zmin:zmax, ymin:ymax, xmin:xmax] = True
        return result.flatten()

    def apply_pattern_transformation_function(self, transform_mat, hup, wup, depth):
        """Apply a transformation matrix on the neurons.

        Args:
            transform_mat (torch.tensor): The transformation matrix.
            hup (int): The height upperbound.
            wup (int): The width upperbound.
            depth (int): The depth.
        """
        big_x_mat = torch.stack(
            [torch.arange(wup)] * hup,
            dim=0,
            dtype=self.neurons.def_dtype,
            device=self.neurons.device,
        )
        big_x_mat = big_x_mat.repeat(depth, 1).flatten()

        big_y_mat = torch.repeat_interleave(
            torch.arange(wup, dtype=self.neurons.def_dtype, device=self.neurons.device),
            hup * depth,
        )

        self.neurons.x = big_x_mat[transform_mat]
        self.neurons.y = big_y_mat[transform_mat]

    def move(self, x=0, y=0, z=0):
        """Move the neurons in the 3D space.

        Args:
            x (int or float): The displacement in x axis.
            y (int or float): The displacement in y axis.
            z (int or float): The displacement in z axis.

        Returns:
            NeuronDimension: The modified `NeuronDimension`.
        """
        self.neurons.x += x
        self.neurons.y += y
        self.neurons.z += z
        return self

    def scale(self, x=1, y=1, z=1):
        """Scale the neuron grid in the 3D space.

        Args:
            x (int or float): The stretch in x axis.
            y (int or float): The stretch in y axis.
            z (int or float): The stretch in z axis.

        Returns:
            NeuronDimension: The modified `NeuronDimension`.
        """
        self.neurons.x *= x
        self.neurons.y *= y
        self.neurons.z *= z
        return self

    def noise(self, x_noise=1, y_noise=1, z_noise=1):
        """Apply random noise to neuron coordinates.

        Args:
            x (int or float): The noise bounds in x axis.
            y (int or float): The noise bounds in y axis.
            z (int or float): The noise bounds in z axis.

        Returns:
            NeuronDimension: The modified `NeuronDimension`.
        """
        self.neurons.x += torch.randint(
            -x_noise, x_noise, self.neurons.size, device=self.neurons.device
        )
        self.neurons.y += torch.randint(
            -y_noise, y_noise, self.neurons.size, device=self.neurons.device
        )
        self.neurons.z += torch.randint(
            -z_noise, z_noise, self.neurons.size, device=self.neurons.device
        )
        return self

    def rotate(self, axis, angle):
        """Rotate the `NeuronGroup` in space.

        Args:
            axis (int): The axis along which the rotation is made.
            angle (float): The angle to rotate in radians.

        Returns:
            NeuronDimension: The modified `NeuronDimension`.
        """
        rotation = rotation_matrix(axis, angle, self.neurons.def_dtype)
        self.neurons.x, self.neurons.y, self.neurons.z = torch.matmul(
            rotation, torch.stack([self.neurons.x, self.neurons.y, self.neurons.z])
        )
        return self

    def stretch_to_equal_size(self, target_neurons):
        """Stretch to `NeuronGroup` to match a target `NeuronGroup` dimensions.

        Args:
            target_neurons (NeuronGroup): The target `NeuronGroup`.
        """
        if hasattr(target_neurons, "width") and self.neurons.width > 0:
            x_stretch = target_neurons.width / self.neurons.width
            self.neurons.x *= x_stretch
        if hasattr(target_neurons, "height") and self.neurons.height > 0:
            y_stretch = target_neurons.height / self.neurons.height
            self.neurons.y *= y_stretch
        if hasattr(target_neurons, "depth") and self.neurons.depth > 0:
            z_stretch = target_neurons.depth / self.neurons.depth
            self.neurons.z *= z_stretch

    def initialize(self, neurons):
        super().initialize(neurons)

        self.width = self.parameter("width", 1, neurons)
        self.height = self.parameter("height", 1, neurons)
        self.depth = self.parameter("depth", 1, neurons)

        for pg in self.parameter("input_patterns", torch.tensor([]), neurons):
            dim = pg.size()
            if len(dim) > 0:
                self.height = max(self.height, dim[0])
            if len(dim) > 1:
                self.width = max(self.width, dim[1])
            if len(dim) > 2:
                self.depth = max(self.depth, dim[0])
                self.height = max(self.height, dim[1])
                self.width = max(self.width, dim[2])

        self.neurons = neurons

        neurons.shape = self
        neurons.width = self.width
        neurons.height = self.height
        neurons.depth = self.depth

        neurons.size = self.width * self.height * self.depth

        if neurons.network.index_neurons:
            self.set_position(self.width, self.height, self.depth)

            if self.parameter("centered", True, neurons):
                self.move(
                    -(self.width - 1) / 2, -(self.height - 1) / 2, -(self.depth - 1) / 2
                )
