# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:15:37 2020

@author: 35732
"""
import configparser
import re

def remove_BOM(config_path):#去掉配置文件开头的BOM字节
    content = open(config_path,encoding="utf-8").read()
    content = re.sub(r"\xfe\xff","", content)
    content = re.sub(r"\ufeff","", content)
    content = re.sub(r"\xff\xfe","", content)
    content = re.sub(r"\xef\xbb\xbf","", content)
    open(config_path, 'w',encoding="utf-8").write(content)

class Dictionary:
    def __init__(self, path):
        remove_BOM(path)
        cp = configparser.RawConfigParser()
        cp.read(path, 'UTF-8')
        self.replace_word = eval(cp.get("SYNTHESIS_EXTRACTOR", 'replace_word'))
        self.alloy_to_replace = eval(cp.get("SYNTHESIS_EXTRACTOR", 'alloy_to_replace'))
        self.paras_to_replace = eval(cp.get("SYNTHESIS_EXTRACTOR", 'paras_to_replace'))
        self.alloy_writing_type = eval(cp.get("SYNTHESIS_EXTRACTOR", 'alloy_writing_type'))
        self.alloy_blank_type = eval(cp.get("SYNTHESIS_EXTRACTOR", 'alloy_blank_type'))
        self.prop_writing_type = eval(cp.get("SYNTHESIS_EXTRACTOR", 'prop_writing_type'))
        self.value_wt = eval(cp.get("SYNTHESIS_EXTRACTOR", 'value_wt'))
        self.other_phase = eval(cp.get("SYNTHESIS_EXTRACTOR", 'other_phase'))
        self.unit_replace = eval(cp.get("SYNTHESIS_EXTRACTOR", 'unit_replace'))
        self.no_unit_para = eval(cp.get("SYNTHESIS_EXTRACTOR", 'no_unit_para'))
        self.other_quality = eval(cp.get("SYNTHESIS_EXTRACTOR", 'other_quality'))
        self.table_alloy_to_replace = eval(cp.get("SYNTHESIS_EXTRACTOR", 'table_alloy_to_replace'))
        self.table_prop_pattern = eval(cp.get("SYNTHESIS_EXTRACTOR", 'table_prop_pattern'))
        self.table_e_pattern = eval(cp.get("SYNTHESIS_EXTRACTOR", 'table_e_pattern'))
        self.table_ratio_pattern = eval(cp.get("SYNTHESIS_EXTRACTOR", 'table_ratio_pattern'))
        self.table_units = eval(cp.get("SYNTHESIS_EXTRACTOR", 'table_units'))
        self.ele_list = eval(cp.get("SYNTHESIS_EXTRACTOR", 'ele_list'))
        self.table_number_pattern = eval(cp.get("SYNTHESIS_EXTRACTOR", 'table_number_pattern'))
        self.unit_pattern_table = eval(cp.get("SYNTHESIS_EXTRACTOR", 'unit_pattern_table'))
        self.ele_to_abr = eval(cp.get("SYNTHESIS_EXTRACTOR", 'ele_to_abr'))
        self.table_prop_pattern_words = eval(cp.get("SYNTHESIS_EXTRACTOR", 'table_prop_pattern_words'))
        self.unit_list = eval(cp.get("SYNTHESIS_EXTRACTOR", 'unit_list'))
        self.symbols = eval(cp.get("SYNTHESIS_EXTRACTOR", 'symbols'))
        self.replace_items = eval(cp.get("SYNTHESIS_EXTRACTOR", 'replace_items'))
        self.ele_list = eval(cp.get("SYNTHESIS_EXTRACTOR", 'ele_list'))
        self.units = eval(cp.get("SYNTHESIS_EXTRACTOR", 'units'))
        self.key_rela_name = eval(cp.get("SYNTHESIS_EXTRACTOR", 'key_rela_name'))
