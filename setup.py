#!/usr/bin/env python
import patsylearn

from os.path import exists
from setuptools import setup

setup(name='patsylearn',
      version=patsylearn.__version__,
      description='Scikit-lean Patsy adaptor',
      url='http://github.com/amueller/patsylearn/',
      maintainer='Andreas Mueller',
      maintainer_email='t3kcit@gmail.com',
      license='BSD',
      keywords='scikit-learn patsy formula machine-learning',
      packages=['patsylearn'],
      long_description=(open('README.md').read() if exists('README.rst') else
                        ''))
