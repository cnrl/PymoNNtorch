import copy

from pymonntorch.NetworkCore.Base import *


class SynapseGroup(NetworkObject):
    def __init__(self, src, dst, net, tag=None, behavior={}):

        if type(src) is str:
            src = net[src, 0]

        if type(dst) is str:
            dst = net[dst, 0]

        if tag is None and net is not None:
            tag = "SynapseGroup_" + str(len(net.SynapseGroups) + 1)

        super().__init__(tag, net, behavior, net.device)
        self.add_tag("syn")

        if len(src.tags) > 0 and len(dst.tags) > 0:
            self.add_tag(src.tags[0] + " => " + dst.tags[0])

        if net is not None:
            net.SynapseGroups.append(self)
            setattr(net, self.tags[0], self)

        self.recording = True

        self.src = src
        self.dst = dst
        self.enabled = True
        self.group_weighting = 1

    def __repr__(self):
        result = (
            "SynapseGroup"
            + str(self.tags)
            + "(D"
            + str(self.dst.size)
            + "xS"
            + str(self.src.size)
            + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k]) + ","
        return result + "}"

    def set_var(self, key, value):
        setattr(self, key, value)
        return self

    def get_synapse_mat_dim(self):
        return self.dst.size, self.src.size

    def get_random_synapse_mat_fixed(self, min_number_of_synapses=0):
        dim = self.get_synapse_mat_dim()
        result = torch.zeros(dim, device=self.device)
        if min_number_of_synapses != 0:
            for i in range(dim[0]):
                synapses = torch.randperm(dim[1], device=self.device)[
                    :min_number_of_synapses
                ]
                result[i, synapses] = torch.rand(len(synapses), device=self.device)
        return result

    def get_synapse_mat(
        self,
        mode="zeros()",
        scale=None,
        density=None,
        only_enabled=True,
        clone_along_first_axis=False,
        plot=False,
        kwargs={},
    ):  # mode in ['zeros', 'zeros()', 'ones', 'ones()', 'uniform(...)', 'lognormal(...)', 'normal(...)']
        result = self._get_mat(
            mode=mode,
            dim=(self.get_synapse_mat_dim()),
            scale=scale,
            density=density,
            plot=plot,
            kwargs=kwargs,
        )

        if clone_along_first_axis:
            result = (
                torch.cat([result[0] for _ in range(self.get_synapse_mat_dim()[0])])
                .reshape(self.get_synapse_mat_dim()[0], *result[0].shape)
                .to(self.device)
            )

        if only_enabled:
            result *= self.enabled

        return result

    def get_synapse_group_size_factor(self, synapse_group, synapse_type):
        total_weighting = 0
        for s in synapse_group.dst.afferent_synapses[synapse_type]:
            total_weighting += s.group_weighting

        total = 0
        for s in synapse_group.dst.afferent_synapses[synapse_type]:
            total += s.src.size * s.src.group_weighting

        return (
            total_weighting
            / total
            * synapse_group.src.size
            * synapse_group.group_weighting
        )

    def get_distance_mat(self, radius, src_x=None, src_y=None, dst_x=None, dst_y=None):
        if src_x is None:
            src_x = self.src.x
        if src_y is None:
            src_y = self.src.y
        if dst_x is None:
            dst_x = self.dst.x
        if dst_y is None:
            dst_y = self.dst.y

        result_syn_mat = torch.zeros((len(dst_x), len(src_x)), device=self.device)

        for d_n in range(len(dst_x)):
            dx = torch.abs(src_x - dst_x[d_n])
            dy = torch.abs(src_y - dst_y[d_n])

            dist = torch.sqrt(dx * dx + dy * dy)
            inv_dist = torch.clamp(radius - dist, 0.0, None)
            inv_dist /= torch.max(inv_dist)

            result_syn_mat[d_n] = inv_dist

        return result_syn_mat

    def get_ring_mat(
        self, radius, inner_exp, src_x=None, src_y=None, dst_x=None, dst_y=None
    ):
        dm = self.get_distance_mat(radius, src_x, src_y, dst_x, dst_y)
        ring = torch.clamp(dm - torch.pow(dm, inner_exp) * 1.5, 0.0, None)
        return ring / torch.max(ring)

    def get_max_receptive_field_size(self):
        max_dx = 1
        max_dy = 1
        max_dz = 1

        for i in range(self.dst.size):
            if type(self.enabled) is torch.tensor:
                mask = self.enabled[i]
            else:
                mask = self.enabled

            if torch.sum(mask) > 0:
                x = self.dst.x[i]
                y = self.dst.y[i]
                z = self.dst.z[i]

                sx_v = self.src.x[mask]
                sy_v = self.src.y[mask]
                sz_v = self.src.z[mask]

                max_dx = torch.maximum(torch.max(torch.abs(x - sx_v)), max_dx)
                max_dy = torch.maximum(torch.max(torch.abs(y - sy_v)), max_dy)
                max_dz = torch.maximum(torch.max(torch.abs(z - sz_v)), max_dz)

        return max_dx, max_dy, max_dz

    def get_sub_synapse_group(self, src_mask, dst_mask):
        result = SynapseGroup(
            self.src.subGroup(src_mask),
            self.dst.subGroup(dst_mask),
            net=None,
            behavior={},
        )

        # partition enabled update
        if type(self.enabled) is torch.tensor:
            mat_mask = dst_mask[:, None] * src_mask[None, :]
            result.enabled = (
                self.enabled[mat_mask].copy().reshape(result.get_synapse_mat_dim())
            )

        # copy al attributes
        sgd = self.__dict__
        for key in sgd:
            if key == "behavior":
                for k in self.behavior:
                    result.behavior[k] = copy.copy(self.behavior[k])
            elif key not in ["src", "dst", "enabled", "_mat_eval_dict"]:
                setattr(result, key, copy.copy(sgd[key]))

        return result
