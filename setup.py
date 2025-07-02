# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:58:57 2022

@author: joverbee
"""

from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
   name='pyEELSMODEL',
   version='1.0.4',
   author='Jo Verbeeck and Daen Jannis',
   author_email='jo.verbeeck@uantwerpen.be',
   package_dir={'': '.'},
   packages=find_packages(),
   url='https://github.com/joverbee/pyEELSMODEL',
   license='LICENSE.txt',
   description='a model based quantification library for electron energy loss spectroscopy',
   long_description=long_description,
   long_description_content_type='text/markdown',
   install_requires=[
       "Django >= 1.1.1",
       "h5py",
       "pqdm",
       "numpy>=1.20.0",
       "matplotlib",
       "scipy",
       "tqdm",
       "fpdf",
       "pyqt5",
       "pytest",
       "jupyter",
       "requests",
       "quadprog"
   ],
   setup_requires=['flake8'],
   test_requires=['pytest'],
   package_data={"pyEELSMODEL": ["element_info.hdf5"]},
   # exclude_package_data={"pyEELSMODEL.database.Zhang": ["*.hdf5"]},
   exclude_package_data={'pyEELSMODEL': ['database/Zhang/*.hdf5',]},

)