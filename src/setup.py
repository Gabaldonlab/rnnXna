#!/usr/bin/env python


import io
import os
import sys
from setuptools import find_packages, setup, Command



NAME = 'rnnxna'
DESCRIPTION = 'Build RNN Models for DNA/RNA Sequences.'
URL = 'https://github.com/Gabaldonlab/rnnXna'
EMAIL = 'ah.hafez@gmail.com'
AUTHOR = 'A. Hafez'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = [
     'numpy',
     'tensorflow'
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    #package_data={'rnnxna': ['model_config/*']},

    entry_points={ ## we might not need this :: we could use conda bin ??
         'console_scripts': ['rnnXna=rnnxna:rnnxna.rnnXnaMain'],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
