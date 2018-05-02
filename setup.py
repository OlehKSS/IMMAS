#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
   name='immas',
   version='1.0',
   description='Intelligent Mammogram Mass Analysis and Segmentation',
   author='MAIA',
   packages=find_packages(),  # finds all the packages in the module
   install_requires=['numpy', 
   'opencv-python', 
   'PyWavelets', 
   'scipy',
   'scikit-learn',
   'pandas'], # external packages as dependencies
)
