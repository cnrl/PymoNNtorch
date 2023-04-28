#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy>=1.21.0', 'torch>=1.11.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Computational Neuroscience Research Laboratory",
    author_email='ashenatena@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="PymoNNtorch is a Pytorch version of PymoNNto",
    entry_points={
        'console_scripts': [
            'pymonntorch=pymonntorch.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pymonntorch',
    name='pymonntorch',
    packages=find_packages(include=['pymonntorch', 'pymonntorch.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/cnrl/PymoNNtorch',
    version='0.1.0',
    zip_safe=False,
)
