# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:58:57 2022

@author: joverbee
"""

from setuptools import setup, find_packages

setup(
   name='pyEELSMODEL',
   version='0.0.1',
   author='Jo Verbeeck and Daen Jannis',
   author_email='jo.verbeeck@uantwerpen.be',
   package_dir={'': '.'},
   packages=find_packages(),
   #['pyEELSMODEL','pyEELSMODEL.core','pyEELSMODEL.dmread','pyEELSMODEL.components','pyEELSMODEL.fitters','pyEELSMODEL.io','pyEELSMODEL.misc','pyEELSMODEL.operators','pyEELSMODEL.tests','pyEELSMODEL.components.CLedge','pyEELSMODEL.components.MScatter'],
   url='',
   license='LICENSE.txt',
   description='a model based quantification library for electron energy loss spectroscopy',
   long_description=open('README.md').read(),
   install_requires=[
       "Django >= 1.1.1",
       "h5py",
       "pqdm",
       "numpy>=1.20.0",
       "matplotlib",
       "scipy",
       "tqdm",
       "fpdf"
   ],
   setup_requires=['flake8'],
   test_requires=['pytest'],



)