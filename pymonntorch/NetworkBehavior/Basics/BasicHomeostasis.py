import torch

from pymonntorch.NetworkCore.Behavior import Behavior
from pymonntorch.utils import check_is_torch_tensor


class InstantHomeostasis(Behavior):
    def set_threshold_boundaries(self, th, gap_percent):
        self.th = th

        if th is not None:
            has_min = False
            has_max = False
            if self.min_th is None:
                self.min_th = th
                has_min = True
            if self.max_th is None:
                self.max_th = th
                has_max = True

            if gap_percent is not None:
                if has_min:
                    self.min_th -= self.min_th / 100 * gap_percent
                    check_is_torch_tensor(
                        self.min_th, device=self.device, dtype=torch.float32
                    )
                if has_max:
                    self.max_th += self.max_th / 100 * gap_percent
                    check_is_torch_tensor(
                        self.max_th, device=self.device, dtype=torch.float32
                    )

    def get_measurement_param(self, object):
        if self.compiled_mp is None:
            self.compiled_mp = compile(self.measurement_param, "<string>", "eval")

        result = check_is_torch_tensor(
            eval(self.compiled_mp), device=object.device, dtype=torch.float32
        )

        result.clamp_(self.measurement_min, self.measurement_max)
        return result

    def get_target_adjustment(self, measure):
        greater = (measure > self.max_th) * (-self.dec)
        smaller = (measure < self.min_th) * self.inc

        if self.distance_sensitive:
            greater *= measure - self.max_th
            smaller *= self.min_th - measure

        return (greater + smaller) * self.adj_strength

    def adjust_target(self, object, adj):
        val = check_is_torch_tensor(
            getattr(object, self.adjustment_param),
            device=object.device,
            dtype=torch.float32,
        )
        val.add_(adj)

        val.clamp_(self.target_clip_min, self.target_clip_max)

        setattr(object, self.adjustment_param, val)

    def set_variables(self, object):
        super().set_variables(object)

        self.compiled_mp = None

        self.min_th = self.get_init_attr(
            "min_th", None, object
        )  # minimum threshold for measurement param
        self.max_th = self.get_init_attr(
            "max_th", None, object
        )  # maximum threshold for measurement param

        self.set_threshold_boundaries(
            self.get_init_attr(
                "threshold", None, object=object
            ),  # threshold for measurement param (min=max=th)
            self.get_init_attr(
                "gap_percent", None, object=object
            ),  # min max gap is created via a percentage from th (additional param for th)
        )

        self.distance_sensitive = self.get_init_attr(
            "distance_sensitive", True, object
        )  # stronger adjustment when value is further away from optimum

        self.inc = check_is_torch_tensor(
            self.get_init_attr("inc", 1.0, object),  # increase factor
            device=object.device,
            dtype=torch.float32,
        )
        self.dec = check_is_torch_tensor(
            self.get_init_attr("dec", 1.0, object),  # decrease factor
            device=object.device,
            dtype=torch.float32,
        )

        self.adj_strength = check_is_torch_tensor(
            self.get_init_attr("adj_strength", 1.0, object),  # change factor
            device=object.device,
            dtype=torch.float32,
        )

        self.adjustment_param = self.get_init_attr(
            "adjustment_param", None, object
        )  # name of object target attribute

        self.measurement_param = self.get_init_attr(
            "measurement_param", None, object
        )  # name of parameter to be measured

        self.measurement_min = check_is_torch_tensor(
            self.get_init_attr(
                "measurement_min", None, object
            ),  # minimum value which can be measured (below=0)
            device=object.device,
            dtype=torch.float32,
        )
        self.measurement_max = check_is_torch_tensor(
            self.get_init_attr(
                "measurement_max", None, object
            ),  # maximum value which can be measured (above=max)
            device=object.device,
            dtype=torch.float32,
        )

        self.target_clip_min = check_is_torch_tensor(
            self.get_init_attr("target_clip_min", None, object),  # target clip min
            device=object.device,
            dtype=torch.float32,
        )
        self.target_clip_max = check_is_torch_tensor(
            self.get_init_attr("target_clip_max", None, object),  # target clip max
            device=object.device,
            dtype=torch.float32,
        )

    def new_iteration(self, object):
        measure = self.get_measurement_param(object)
        self.adj = self.get_target_adjustment(measure)
        self.adjust_target(object, self.adj)


class TimeIntegratedHomeostasis(InstantHomeostasis):
    def get_measurement_param(self, object):
        val = super().get_measurement_param(object)

        self.average = (self.average * self.integration_length + val) / (
            self.integration_length + 1
        )
        return self.average

    def set_variables(self, object):
        super().set_variables(object)

        self.integration_length = check_is_torch_tensor(
            self.get_init_attr("integration_length", 1, object),
            device=object.device,
            dtype=torch.float32,
        )
        self.average = check_is_torch_tensor(
            self.get_init_attr("init_avg", (self.min_th + self.max_th) / 2, object),
            device=object.device,
            dtype=torch.float32,
        )
