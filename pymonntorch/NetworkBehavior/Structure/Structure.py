import torch
from pymonntorch.NetworkCore.Behavior import Behavior


def vec_to_mat(vec, repeat_count):
    return torch.stack([vec] * repeat_count, dim=0)


def vec_to_mat_transposed(vec, repeat_count):
    return torch.stack([vec] * repeat_count, dim=1)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = torch.tensor(axis, dtype=torch.float)
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_squared_dim(n_neurons, depth=1):
    """
    Get the squared dimension of a square matrix that can hold n_neurons
    neurons, with depth layers.
    """
    divider = int(torch.trunc(torch.sqrt(torch.tensor(n_neurons))))

    while divider > 1 and n_neurons % divider != 0:
        divider -= 1

    width = divider
    height = n_neurons // divider if divider > 0 else 0

    print(f"Set population size to {depth}x{height}x{width}.")

    return NeuronDimension(width, height, depth)


class NeuronDimension(Behavior):

    set_variables_on_init = True

    def set_position(self, width, height, depth):
        self.neurons.x = torch.arange(0, self.neurons.size) % width
        self.neurons.y = torch.arange(0, self.neurons.size) // width % height
        self.neurons.z = torch.arange(0, self.neurons.size) // (width * height) % depth

    def get_area_mask(self, xmin=0, xmax=-1, ymin=0, ymax=-1, zmin=0, zmax=-1):
        if xmax < 1:
            xmax = self.width - xmax
        if ymax < 1:
            ymax = self.height - ymax
        if zmax < 1:
            zmax = self.depth - zmax

        result = torch.zeros(self.depth, self.height, self.depth, dtype=torch.bool, device=self.neurons.device)
        result[zmin:zmax, ymin:ymax, xmin:xmax] = True
        return result.flatten()

    def apply_pattern_transformation_function(self, transform_mat, hup, wup, depth):
        big_x_mat = torch.stack([torch.arange(wup)] * hup, dim=0)
        big_x_mat = big_x_mat.repeat(depth, 1).flatten()

        big_y_mat = torch.repeat_interleave(torch.arange(wup), hup * depth)

        self.neurons.x = big_x_mat[transform_mat]
        self.neurons.y = big_y_mat[transform_mat]

    def move(self, x=0, y=0, z=0):
        self.neurons.x += x
        self.neurons.y += y
        self.neurons.z += z
        return self

    def scale(self, x=1, y=1, z=1):
        self.neurons.x *= x
        self.neurons.y *= y
        self.neurons.z *= z
        return self

    def noise(self, x_noise=1, y_noise=1, z_noise=1, centered=True):
        self.neurons.x += torch.randint(-x_noise, x_noise, self.neurons.size, device=self.neurons.device)
        self.neurons.y += torch.randint(-y_noise, y_noise, self.neurons.size, device=self.neurons.device)
        self.neurons.z += torch.randint(-z_noise, z_noise, self.neurons.size, device=self.neurons.device)
        return self

    def rotate(self, axis, angle):
        rotation = rotation_matrix(axis, angle)
        self.neurons.x, self.neurons.y, self.neurons.z = torch.matmul(
            rotation, torch.stack([self.neurons.x, self.neurons.y, self.neurons.z])
            )
        return self

    def stretch_to_equal_size(self, target_neurons):
        if hasattr(target_neurons, "width") and self.neurons.width > 0:
            x_stretch = target_neurons.width / self.neurons.width
            self.neurons.x *= x_stretch
        if hasattr(target_neurons, "height") and self.neurons.height > 0:
            y_stretch = target_neurons.height / self.neurons.height
            self.neurons.y *= y_stretch
        if hasattr(target_neurons, "depth") and self.neurons.depth > 0:
            z_stretch = target_neurons.depth / self.neurons.depth
            self.neurons.z *= z_stretch

    def set_variables(self, object):
        self.width = self.get_init_attr("width", 1, object)
        self.height = self.get_init_attr("height", 1, object)
        self.depth = self.get_init_attr("depth", 1, object)

        for pg in self.get_init_attr('input_patterns', torch.tensor([]), object):
            dim = pg.size()
            if len(dim) > 0:
                self.height = max(self.height, dim[0])
            if len(dim) > 1:
                self.width = max(self.width, dim[1])
            if len(dim) > 2:
                self.depth = max(self.depth, dim[0])
                self.height = max(self.height, dim[1])
                self.width = max(self.width, dim[2])

        self.neurons = object

        object.shape = self
        object.width = self.width
        object.height = self.height
        object.depth = self.depth

        object.size=self.width*self.height*self.depth

        self.set_positions(self.width, self.height, self.depth)

        if self.get_init_attr('centered', True, object):
            self.move(-(self.width-1)/2,-(self.height-1)/2,-(self.depth-1)/2)

    def new_iteration(self, object):
        return
