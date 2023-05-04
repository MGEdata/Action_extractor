# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:15:37 2020

@author: 35732
"""
import configparser


class Dictionary:
    def __init__(self, path):
        cp = configparser.RawConfigParser()
        cp.read(path, 'UTF-8')
        self.bef_existed = eval(cp.get("DICTIONARY", 'bef_existed'))

