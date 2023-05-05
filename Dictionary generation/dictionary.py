# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:15:37 2020

@author: wwr
"""
import configparser

class Dictionary:
    def __init__(self, path):
        cp = configparser.RawConfigParser()
        cp.read(path, 'UTF-8')
        self.bef_existed = eval(cp.get("DICTIONARY", 'bef_existed'))
        self.aft_rule = eval(cp.get("DICTIONARY", 'aft_rule'))
        self.bef_rule = eval(cp.get("DICTIONARY", 'bef_rule'))
        self.chunk_limit = eval(cp.get("DICTIONARY", 'chunk_limit'))
        self.error_chunks = eval(cp.get("DICTIONARY", 'error_chunks'))
        self.symbols = eval(cp.get("DICTIONARY", 'symbols'))
        self.unit_list = eval(cp.get("DICTIONARY", 'unit_list'))

