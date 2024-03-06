# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:46:09 2021

@author: joverbee
"""
import logging
logger = logging.getLogger(__name__)


class Operator():
    """
    Operator class from which all other operations on the spectrum class are
    derived
    """
    def __init__(self):
        self.name = 'Operator'
