===========
PymoNNtorch
===========


.. image:: https://img.shields.io/pypi/v/pymonntorch.svg
        :target: https://pypi.python.org/pypi/pymonntorch

.. image:: https://img.shields.io/travis/cnrl/pymonntorch.svg
        :target: https://travis-ci.com/cnrl/pymonntorch

.. image:: https://readthedocs.org/projects/pymonntorch/badge/?version=latest
        :target: https://pymonntorch.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




PymoNNtorch is a Pytorch adapted version of PymoNNto_.

.. _PymoNNto: https://github.com/trieschlab/PymoNNto


* Free software: MIT license
* Documentation: https://pymonntorch.readthedocs.io.


Features
--------

* Use `torch` tensors and Pytorch-like syntax to create a spiking neural network (SNN).
* Simulate an SNN on CPU or GPU.
* Define dynamics of SNN components as `Behavior` modules.
* Control over the order of applying different `Behavior`s in each simulation time step.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.
It changes the codebase of PymoNNto_ to use `torch` rather than `numpy` and `tensorflow numpy`.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _PymoNNto: https://github.com/trieschlab/PymoNNto
