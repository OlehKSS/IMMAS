#!/usr/bin/env python

from setuptools import setup

setup(
   name='immas',
   version='1.0',
   description='Intelligent Mammogram Mass Analysis and Segmentation',
   author='MAIA',
   packages=['immas'],  #same as name
   install_requires=['numpy', 'opencv-python'], #external packages as dependencies
)
