import torch

from pymonntorch.NetworkCore.Behavior import Behavior
from pymonntorch.utils import check_is_torch_tensor


class InstantHomeostasis(Behavior):
    def __init__(
        self,
        *args,
        min_th=None,
        max_th=None,
        threshold=None,
        gap_percent=None,
        distance_sensitive=True,
        inc=1.0,
        dec=1.0,
        adj_strength=1.0,
        adjustment_param=None,
        measurement_param=None,
        measurement_min=None,
        measurement_max=None,
        target_clip_min=None,
        target_clip_max=None,
        **kwargs
    ):
        super().__init__(
            *args,
            min_th=min_th,
            max_th=max_th,
            threshold=threshold,
            gap_percent=gap_percent,
            distance_sensitive=distance_sensitive,
            inc=inc,
            dec=dec,
            adj_strength=adj_strength,
            adjustment_param=adjustment_param,
            measurement_param=measurement_param,
            measurement_min=measurement_min,
            measurement_max=measurement_max,
            target_clip_min=target_clip_min,
            target_clip_max=target_clip_max,
            **kwargs
        )

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
                        self.min_th, device=self.device, dtype=self.dtype
                    )
                if has_max:
                    self.max_th += self.max_th / 100 * gap_percent
                    check_is_torch_tensor(
                        self.max_th, device=self.device, dtype=self.dtype
                    )

    def get_measurement_param(self, object):
        if self.compiled_mp is None:
            self.compiled_mp = compile(self.measurement_param, "<string>", "eval")

        result = check_is_torch_tensor(
            eval(self.compiled_mp), device=object.device, dtype=self.dtype
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
            dtype=self.dtype,
        )
        val.add_(adj)

        val.clamp_(self.target_clip_min, self.target_clip_max)

        setattr(object, self.adjustment_param, val)

    def initialize(self, object):
        super().initialize(object)

        self.dtype = object.def_dtype

        self.compiled_mp = None

        self.min_th = self.parameter(
            "min_th", None, object
        )  # minimum threshold for measurement param
        self.max_th = self.parameter(
            "max_th", None, object
        )  # maximum threshold for measurement param

        self.set_threshold_boundaries(
            self.parameter(
                "threshold", None, object=object
            ),  # threshold for measurement param (min=max=th)
            self.parameter(
                "gap_percent", None, object=object
            ),  # min max gap is created via a percentage from th (additional param for th)
        )

        self.distance_sensitive = self.parameter(
            "distance_sensitive", True, object
        )  # stronger adjustment when value is further away from optimum

        self.inc = check_is_torch_tensor(
            self.parameter("inc", 1.0, object),  # increase factor
            device=object.device,
            dtype=self.dtype,
        )
        self.dec = check_is_torch_tensor(
            self.parameter("dec", 1.0, object),  # decrease factor
            device=object.device,
            dtype=self.dtype,
        )

        self.adj_strength = check_is_torch_tensor(
            self.parameter("adj_strength", 1.0, object),  # change factor
            device=object.device,
            dtype=self.dtype,
        )

        self.adjustment_param = self.parameter(
            "adjustment_param", None, object
        )  # name of object target attribute

        self.measurement_param = self.parameter(
            "measurement_param", None, object
        )  # name of parameter to be measured

        self.measurement_min = check_is_torch_tensor(
            self.parameter(
                "measurement_min", None, object
            ),  # minimum value which can be measured (below=0)
            device=object.device,
            dtype=self.dtype,
        )
        self.measurement_max = check_is_torch_tensor(
            self.parameter(
                "measurement_max", None, object
            ),  # maximum value which can be measured (above=max)
            device=object.device,
            dtype=self.dtype,
        )

        self.target_clip_min = check_is_torch_tensor(
            self.parameter("target_clip_min", None, object),  # target clip min
            device=object.device,
            dtype=self.dtype,
        )
        self.target_clip_max = check_is_torch_tensor(
            self.parameter("target_clip_max", None, object),  # target clip max
            device=object.device,
            dtype=self.dtype,
        )

    def forward(self, object):
        measure = self.get_measurement_param(object)
        self.adj = self.get_target_adjustment(measure)
        self.adjust_target(object, self.adj)


class TimeIntegratedHomeostasis(InstantHomeostasis):
    def __init__(
        self,
        *args,
        min_th=None,
        max_th=None,
        threshold=None,
        gap_percent=None,
        distance_sensitive=True,
        inc=1.0,
        dec=1.0,
        adj_strength=1.0,
        adjustment_param=None,
        measurement_param=None,
        measurement_min=None,
        measurement_max=None,
        target_clip_min=None,
        target_clip_max=None,
        integration_length=1,
        init_avg=None,
        **kwargs
    ):
        super().__init__(
            *args,
            min_th=min_th,
            max_th=max_th,
            threshold=threshold,
            gap_percent=gap_percent,
            distance_sensitive=distance_sensitive,
            inc=inc,
            dec=dec,
            adj_strength=adj_strength,
            adjustment_param=adjustment_param,
            measurement_param=measurement_param,
            measurement_min=measurement_min,
            measurement_max=measurement_max,
            target_clip_min=target_clip_min,
            target_clip_max=target_clip_max,
            integration_length=integration_length,
            init_avg=init_avg,
            **kwargs
        )

    def get_measurement_param(self, object):
        val = super().get_measurement_param(object)

        self.average = (self.average * self.integration_length + val) / (
            self.integration_length + 1
        )
        return self.average

    def initialize(self, object):
        super().initialize(object)

        self.integration_length = check_is_torch_tensor(
            self.parameter("integration_length", 1, object),
            device=object.device,
            dtype=object.def_dtype,
        )
        self.average = check_is_torch_tensor(
            self.parameter("init_avg", (self.min_th + self.max_th) / 2, object),
            device=object.device,
            dtype=object.def_dtype,
        )
