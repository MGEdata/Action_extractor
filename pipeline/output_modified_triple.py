# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:45:27 2020

@author: wwr
"""

import os
import re
import nltk
from tqdm import tqdm
import openpyxl
import xlrd
from dictionary import Dictionary
from log_wp import LogWp
import json

all_names = list()
topic_names = list()

class TableExtraction:
    def __init__(self, excels_path, c_path):
        self.excels_path = excels_path
        self.c_path = c_path
        self.dict_info = Dictionary(self.c_path)
        self.ele_list = self.dict_info.ele_list
        self.e_pattern = self.dict_info.table_e_pattern
        self.ratio_pattern = self.dict_info.table_ratio_pattern
        self.prop_pattern = self.dict_info.table_prop_pattern
        self.unit_pattern = self.dict_info.unit_pattern_table
        self.number_pattern = self.dict_info.table_number_pattern
        self.ele_to_abr = self.dict_info.ele_to_abr
        self.prop_pattern_words = self.dict_info.table_prop_pattern_words
        self.log_wp = LogWp()

    def mat_process(self, mat):
        mats = re.findall(r"\[\d+\]", str(mat))
        for ma in mats:
            mat = mat.replace(ma,"")
        return mat

    def composition_triple_extraction(self):
        file_list = os.listdir(self.excels_path)
        composition_all = {}
        for excel_path in file_list:
            try:
                file = xlrd.open_workbook(self.excels_path + '/' + excel_path)
                all_material = []
                for sheet_i in range(len(file.sheets())):
                    try:
                        sheet = file.sheet_by_index(sheet_i)
                        topic = sheet.row_values(1)[0]
                        if 'composition' in topic.lower():
                            target_ele_row = []
                            target_ele_col = []
                            search_outcome = []
                            ele_loc = None
                            for line_index in range(2, len(sheet.col_values(0))):
                                search_line = sheet.row_values(line_index)
                                unit_i = 0
                                for unit in search_line:
                                    outcome = re.findall(self.e_pattern, str(unit))
                                    if outcome and str(unit) in self.ele_list:
                                        target_ele_row.append(line_index)
                                        target_ele_col.append(unit_i)
                                        search_outcome.append(unit)
                                    unit_i += 1
                                if search_outcome:
                                    ele_loc = line_index
                                    break
                            if ele_loc:
                                dict_info = Dictionary(self.c_path)
                                alloy_replace = dict_info.table_alloy_to_replace
                                alloy_common_type = dict_info.alloy_writing_type
                                alloy_blank_type = dict_info.alloy_blank_type
                                for alloy_model, replace in alloy_replace.items():
                                    alloy_part = re.findall(alloy_model, str(topic))
                                    for alloy in alloy_part:
                                        find_part = re.findall(replace[0], str(alloy))
                                        alloy_out = alloy.replace(find_part[0], replace[1])
                                        topic = topic.replace(alloy, alloy_out)
                                    outcome_name = list()
                                    topic_tokenize = nltk.word_tokenize(topic)
                                    for word in topic_tokenize:
                                        for pattern_1 in alloy_common_type:
                                            outcome_common = re.findall(pattern_1, str(word))
                                            if outcome_common:
                                                outcome_name.append(word)
                                                break
                                    for pattern_2 in alloy_blank_type:
                                        outcome_blank = re.findall(pattern_2, str(topic))
                                        if outcome_blank and outcome_blank[0] not in outcome_name:
                                            outcome_name.append(outcome_blank[0])
                                            break
                                len_col = len(sheet.row_values(3))
                                alloy_name_col = None
                                alloy_name_search = []
                                if len_col <= 3:
                                    for col_i in range(len_col):
                                        col_info = sheet.col_values(col_i)
                                        if col_i == 0:
                                            col_info = sheet.col_values(col_i)[2:]
                                        if col_info:
                                            for cell in col_info:
                                                for pattern_1 in alloy_common_type:
                                                    outcome_common = re.findall(pattern_1, str(cell))
                                                    if outcome_common:
                                                        alloy_name_col = col_i
                                                        alloy_name_search.append(col_i)
                                                for pattern_2 in alloy_blank_type:
                                                    outcome_blank = re.findall(pattern_2, str(cell))
                                                    if outcome_blank:
                                                        alloy_name_col = col_i
                                                        alloy_name_search.append(col_i)
                                else:
                                    for col_i in range(3):
                                        col_info = sheet.col_values(col_i)
                                        if col_i == 0:
                                            col_info = sheet.col_values(col_i)[2:]
                                        if col_info:
                                            for cell in col_info:
                                                for pattern_1 in alloy_common_type:
                                                    outcome_common = re.findall(pattern_1, str(cell))
                                                    if outcome_common:
                                                        alloy_name_col = col_i
                                                        alloy_name_search.append(col_i)
                                                for pattern_2 in alloy_blank_type:
                                                    outcome_blank = re.findall(pattern_2, str(cell))
                                                    if outcome_blank:
                                                        alloy_name_col = col_i
                                                        alloy_name_search.append(col_i)
                                if not alloy_name_search:
                                    alloy_name_col = 0
                                else:
                                    alloy_name_col = alloy_name_search[0]
                                first_col = sheet.col_values(0)
                                ele_first = []
                                for unit in first_col:
                                    firstcol_search = re.findall(self.e_pattern, str(unit))
                                    if firstcol_search:
                                        ele_first.append(unit)
                                if len(ele_first) <= 2:
                                    if len(first_col) > 4:
                                        e_search = re.findall(self.e_pattern, str(sheet.col_values(0)[ele_loc]))
                                        if e_search and outcome_name and len(outcome_name) == 1:
                                            for index_row in range(ele_loc + 1, len(first_col)):
                                                composition_single = {}
                                                composition_single['material'] = outcome_name[0].replace('~', ' ')
                                                composition_single['doi'] = first_col[0]
                                                ratio_find_topic = re.findall(self.ratio_pattern, str(topic))
                                                ratio_find_col = re.findall(self.ratio_pattern,
                                                                            str(first_col[index_row]))
                                                for table_head in sheet.row_values(2):
                                                    ratio_find_head = re.findall(self.ratio_pattern,
                                                                                 str(table_head))
                                                    if ratio_find_head:
                                                        composition_single['percentage'] = ratio_find_head[0]
                                                        break
                                                if ratio_find_topic:
                                                    composition_single['percentage'] = ratio_find_topic[0]
                                                elif ratio_find_col:
                                                    composition_single['percentage'] = ratio_find_col[0]
                                                for ele_index in range(len(sheet.row_values(2))):
                                                    ele_name = sheet.row_values(ele_loc)[ele_index]
                                                    if ele_name in tuple(self.ele_to_abr.keys()):
                                                        ele_name = self.ele_to_abr[ele_name]
                                                    number = sheet.row_values(index_row)[ele_index]
                                                    composition_single[ele_name] = number

                                                all_material.append(composition_single)
                                        if not e_search:
                                            for index_row in range(ele_loc + 1, len(first_col)):
                                                if first_col[index_row]:
                                                    composition_single = {}
                                                    name_col = sheet.col_values(alloy_name_col)
                                                    if outcome_name and len(
                                                            outcome_name) == 1 and not alloy_name_search:
                                                        composition_single['material'] = outcome_name[0].replace(
                                                            '~',
                                                            ' ')
                                                    else:
                                                        composition_single['material'] = name_col[index_row]
                                                    composition_single['doi'] = first_col[0]
                                                    ratio_find_topic = re.findall(self.ratio_pattern, str(topic))
                                                    ratio_find_col = re.findall(self.ratio_pattern,
                                                                                str(first_col[index_row]))
                                                    for table_head in sheet.row_values(2):
                                                        ratio_find_head = re.findall(self.ratio_pattern,
                                                                                     str(table_head))
                                                        if ratio_find_head:
                                                            composition_single['percentage'] = ratio_find_head[0]
                                                            break
                                                    if ratio_find_topic:
                                                        composition_single['percentage'] = ratio_find_topic[0]
                                                    elif ratio_find_col:
                                                        composition_single['percentage'] = ratio_find_col[0]
                                                    ratio_find_unit = re.findall(self.ratio_pattern,
                                                                                 str(first_col[index_row]))
                                                    if ratio_find_unit:
                                                        composition_single['percentage'] = ratio_find_unit[0]
                                                    for ele_index in range(len(sheet.row_values(ele_loc)[1:])):
                                                        ele_name = sheet.row_values(ele_loc)[1:][ele_index]
                                                        if ele_name in tuple(self.ele_to_abr.keys()):
                                                            ele_name = self.ele_to_abr[ele_name]
                                                        number = sheet.row_values(index_row)[ele_index + 1]
                                                        composition_single[ele_name] = number

                                                    all_material.append(composition_single)
                                    else:
                                        composition_single = {}
                                        first_col_1 = sheet.row_values(3)[0]
                                        e_search = re.findall(self.e_pattern, str(sheet.col_values(0)[ele_loc]))
                                        ratio_find_col = re.findall(self.ratio_pattern, str(first_col_1))
                                        for table_head in sheet.row_values(2):
                                            ratio_find_head = re.findall(self.ratio_pattern, str(table_head))
                                            if ratio_find_head:
                                                composition_single['percentage'] = ratio_find_head[0]
                                                break
                                        if ratio_find_col:
                                            composition_single['percentage'] = ratio_find_col[0]
                                        ratio_find_topic = re.findall(self.ratio_pattern, str(topic))
                                        if ratio_find_topic:
                                            composition_single['percentage'] = ratio_find_topic[0]
                                        if outcome_name and e_search:
                                            composition_single['material'] = outcome_name[0].replace('~', ' ')
                                            composition_single['doi'] = first_col[0]
                                            for ele_index in range(len(sheet.row_values(2))):
                                                ele_name = sheet.row_values(ele_loc)[ele_index]
                                                number = sheet.row_values(3)[ele_index]
                                                if ele_name in tuple(self.ele_to_abr.keys()):
                                                    ele_name = self.ele_to_abr[ele_name]
                                                composition_single[ele_name] = number

                                            all_material.append(composition_single)
                                        elif outcome_name and not e_search:
                                            if len(outcome_name) == 1:
                                                composition_single['material'] = outcome_name[0].replace('~', ' ')
                                            else:
                                                composition_single['material'] = sheet.row_values(ele_loc + 1)[
                                                    alloy_name_col]
                                            composition_single['doi'] = first_col[0]
                                            for ele_index in range(len(sheet.row_values(2)[1:])):
                                                ele_name = sheet.row_values(ele_loc)[1:][ele_index]
                                                number = sheet.row_values(3)[1:][ele_index]
                                                if ele_name in tuple(self.ele_to_abr.keys()):
                                                    ele_name = self.ele_to_abr[ele_name]
                                                composition_single[ele_name] = number
                                            all_material.append(composition_single)
                                        elif not outcome_name and not e_search:
                                            composition_single['material'] = sheet.row_values(ele_loc + 1)[
                                                alloy_name_col]
                                            composition_single['doi'] = first_col[0]
                                            m_name = sheet.row_values(ele_loc)[0]
                                            composition_single[m_name] = first_col[3]
                                            for ele_index in range(len(sheet.row_values(2)[1:])):
                                                ele_name = sheet.row_values(ele_loc)[1:][ele_index]
                                                number = sheet.row_values(3)[1:][ele_index]
                                                if ele_name in tuple(self.ele_to_abr.keys()):
                                                    ele_name = self.ele_to_abr[ele_name]
                                                composition_single[ele_name] = number
                                            all_material.append(composition_single)
                                        elif not outcome_name and e_search:
                                            composition_single['material'] = None
                                            composition_single['doi'] = first_col[0]
                                            for ele_index in range(len(sheet.row_values(2))):
                                                ele_name = sheet.row_values(ele_loc)[ele_index]
                                                number = sheet.row_values(3)[ele_index]
                                                if ele_name in tuple(self.ele_to_abr.keys()):
                                                    ele_name = self.ele_to_abr[ele_name]
                                                composition_single[ele_name] = number
                                            all_material.append(composition_single)
                                else:
                                    ele_row = sheet.row_values(ele_loc - 1)
                                    len_elerow = len(ele_row)
                                    for index_col in range(1, len_elerow):
                                        if ele_row[index_col]:
                                            composition_single = {}
                                            if outcome_name and len(outcome_name) == 1 and len_elerow <= 2:
                                                material_name = outcome_name[0].replace('~', ' ')
                                            else:
                                                material_name = ele_row[index_col]
                                            composition_single['material'] = material_name
                                            composition_single['doi'] = first_col[0]
                                            ratio_find_topic = re.findall(self.ratio_pattern, str(topic))
                                            ratio_find_col = re.findall(self.ratio_pattern, str(material_name))
                                            if ratio_find_topic:
                                                composition_single['percentage'] = ratio_find_topic[0]
                                            elif ratio_find_col:
                                                composition_single['percentage'] = ratio_find_col[0]
                                            for ele_index in range(len(sheet.col_values(0)[ele_loc:])):
                                                ele_name = sheet.col_values(0)[ele_loc:][ele_index]
                                                number = sheet.col_values(index_col)[ele_loc + ele_index]
                                                if ele_name in tuple(self.ele_to_abr.keys()):
                                                    ele_name = self.ele_to_abr[ele_name]
                                                composition_single[ele_name] = number
                                            all_material.append(composition_single)
                        # if all_material:
                        #     break
                    except Exception as e:
                        self.log_wp.print_log("%s", str(e))
                        self.log_wp.print_log("An error in the %s of %s!", sheet_i, excel_path)
                if all_material:
                    doi = excel_path.replace(".xlsx", "")
                    doi = doi.replace("/", "-")
                    composition_all[doi] = all_material
            except Exception as e:
                self.log_wp.print_log("can't open this file, name of file is %s", str(excel_path))
                self.log_wp.print_log("Error is %s", str(e))
                self.log_wp.print_log("%s", "--" * 25)
        return composition_all

    def property_info_extraction(self,prop_names,composition):
        file_list = os.listdir(self.excels_path)
        property_all = {}
        number_prop = 0
        K_path = []

        for excel_path in tqdm(file_list):
                try:
                    file = xlrd.open_workbook(self.excels_path + '/' + excel_path)
                    doi = excel_path.replace(".xlsx", "")
                    doi = doi.replace("/", "-")
                    table_mats_name = list()
                    if doi in composition.keys():
                        for info in composition[doi]:
                            if info["material"]:
                                table_mats_name.append(info["material"])
                    all_material = []
                    for sheet_i in range(len(file.sheets())):
                        try:
                            for prop_name in prop_names:
                                sheet = file.sheet_by_index(sheet_i)
                                topic = sheet.row_values(1)[0]
                                first_col = sheet.col_values(0)[2:]
                                # print(first_col)
                                # print(topic)
                                topic_words = nltk.word_tokenize(topic)
                                if "composition" in topic.lower() and any(word in topic.lower() for word in self.prop_pattern_words[prop_name]):
                                    # print(topic)
                                    topic_names.append(excel_path)
                                    search_outcome = []
                                    target_prop_row = []
                                    target_prop_col = []
                                    for line_index in range(2,4,1):
                                        search_line = sheet.row_values(line_index)[1:]
                                        unit_i = 1
                                        for unit in search_line:
                                            outcome_words = None
                                            unit_words = nltk.word_tokenize(str(unit).lower())
                                            # if any(word in str(unit).lower() for word in
                                            #        self.prop_pattern_words[prop_name]):
                                            #     all_names.append(str(unit))

                                            # if any(word in unit_words for word in self.prop_pattern_words[prop_name]) and len(unit_words)<=3:
                                            #     all_names[str(unit)] = excel_path
                                            #     target_prop_row.append(line_index)
                                            #     target_prop_col.append(unit_i)
                                            #     search_outcome.append(unit)

                                            # print(self.prop_pattern[prop_name])
                                            if any(word in str(unit).lower() for word in self.prop_pattern[prop_name]):
                                                # print(unit)
                                                all_names.append(excel_path)
                                                target_prop_row.append(line_index)
                                                target_prop_col.append(unit_i)
                                                search_outcome.append(unit)
                                            unit_i += 1
                                    if search_outcome:
                                        # print(search_outcome)
                                        first_col = sheet.col_values(0)
                                        alloy_replace = Dictionary(self.c_path).table_alloy_to_replace
                                        for alloy_model, replace in alloy_replace.items():
                                            alloy_part = re.findall(alloy_model, str(topic))
                                            for alloy in alloy_part:
                                                find_part = re.findall(replace[0], str(alloy))
                                                alloy_out = alloy.replace(find_part[0], replace[1])
                                                topic = topic.replace(alloy, alloy_out)
                                        alloy_common_type = Dictionary(self.c_path).alloy_writing_type
                                        alloy_blank_type = Dictionary(self.c_path).alloy_blank_type
                                        outcome_name = []
                                        topic_tokenize = nltk.word_tokenize(topic)
                                        for word in topic_tokenize:
                                            for pattern_1 in alloy_common_type:
                                                outcome_common = re.findall(pattern_1, str(word))
                                                if outcome_common:
                                                    outcome_name.append(word)
                                                    break
                                        for pattern_2 in alloy_blank_type:
                                            outcome_blank = re.findall(pattern_2, str(topic))
                                            if outcome_blank:
                                                outcome_name.append(outcome_blank[0])
                                                break
                                        fc_ns = []
                                        for cell in sheet.col_values(0)[1:]:
                                            fc_n = re.findall(self.number_pattern[prop_name], str(cell))
                                            alphabet_search = re.findall("[A-Za-z]", str(cell))
                                            if fc_n and not alphabet_search:
                                                fc_ns.append(cell)
                                        len_col = len(sheet.row_values(3))
                                        alloy_name_col = None
                                        alloy_name_search = []
                                        if len_col <= 3:
                                            for col_i in range(len_col):
                                                col_info = sheet.col_values(col_i)
                                                if col_i == 0:
                                                    col_info = sheet.col_values(col_i)[2:]
                                                if col_info:
                                                    for cell in col_info:
                                                        for pattern_1 in alloy_common_type:
                                                            outcome_common = re.findall(pattern_1, str(cell))
                                                            if outcome_common:
                                                                alloy_name_col = col_i
                                                                alloy_name_search.append(col_i)
                                                        for pattern_2 in alloy_blank_type:
                                                            outcome_blank = re.findall(pattern_2, str(cell))
                                                            if outcome_blank:
                                                                alloy_name_col = col_i
                                                                alloy_name_search.append(col_i)
                                        else:
                                            for col_i in range(3):
                                                col_info = sheet.col_values(col_i)
                                                if col_i == 0:
                                                    col_info = sheet.col_values(col_i)[2:]
                                                if col_info:
                                                    for cell in col_info:
                                                        for pattern_1 in alloy_common_type:
                                                            outcome_common = re.findall(pattern_1, str(cell))
                                                            if outcome_common:
                                                                alloy_name_col = col_i
                                                                alloy_name_search.append(col_i)
                                                        for pattern_2 in alloy_blank_type:
                                                            outcome_blank = re.findall(pattern_2, str(cell))
                                                            if outcome_blank:
                                                                alloy_name_col = col_i
                                                                alloy_name_search.append(col_i)
                                        if not alloy_name_search:
                                            alloy_name_col = 0
                                        else:
                                            alloy_name_col = alloy_name_search[0]
                                        if len(first_col) > 4:
                                            for prop_i in range(len(target_prop_row)):
                                                sub_label = []
                                                curr_col = []
                                                for index_row in range(target_prop_row[prop_i] + 1, len(first_col)):
                                                    unit_search_parts = []
                                                    unit_search_parts.append(topic)
                                                    if len(fc_ns) == 0:
                                                        name_col = sheet.col_values(alloy_name_col)
                                                        material_name = name_col[index_row]
                                                        property_single = {}
                                                        number = sheet.row_values(index_row)[target_prop_col[prop_i]]
                                                        number_inspect = re.findall(self.number_pattern[prop_name],
                                                                                    str(number))

                                                        unit_search_parts.append(first_col[index_row])
                                                        unit_search_parts.append(number)
                                                        for unit in sheet.row_values(target_prop_row[0]):
                                                            unit_search_parts.append(unit)
                                                        for row_s in range(2, target_prop_row[prop_i] + 1):
                                                            unit_search_parts.append(
                                                                sheet.row_values(row_s)[target_prop_col[prop_i]])
                                                        if number_inspect:
                                                            one_info = {}
                                                            for prop_index in range(len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = sheet.row_values(target_prop_row[prop_i])[
                                                                    prop_index]
                                                                number_line_line = sheet.row_values(index_row)[prop_index]
                                                                one_info[prop_name_line] = number_line_line
                                                            curr_col.append(number)
                                                            property_single[prop_name] = number
                                                            property_single['other_info'] = one_info
                                                            property_single['material'] = material_name
                                                            property_single['doi'] = first_col[0]
                                                            if sub_label:
                                                                property_single['child_tag'] = sub_label
                                                            for item in unit_search_parts:
                                                                unit_find = re.findall(self.unit_pattern[prop_name], str(item))
                                                                if unit_find:
                                                                    property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                                    K_path.append(excel_path)
                                                            if 'unit' not in property_single.keys():
                                                                property_single['unit'] = 'no mentioned'
                                                        elif not number_inspect and len(curr_col) != 0:
                                                            property_single['material'] = material_name
                                                            property_single['doi'] = first_col[0]
                                                            property_single[prop_name] = number
                                                            if sub_label:
                                                                property_single['child_tag'] = sub_label
                                                            for item in unit_search_parts:
                                                                unit_find = re.findall(self.unit_pattern[prop_name], str(item))
                                                                if unit_find:
                                                                    property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                                    K_path.append(excel_path)
                                                                    break
                                                            if 'unit' not in property_single.keys():
                                                                property_single['unit'] = 'no mentioned'

                                                            one_info = {}
                                                            for prop_index in range(
                                                                    len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = \
                                                                    sheet.row_values(target_prop_row[prop_i])[
                                                                        prop_index]
                                                                number_line_line = sheet.row_values(index_row)[
                                                                    prop_index]
                                                                one_info[prop_name_line] = number_line_line

                                                            property_single['other_info'] = one_info
                                                        elif not number_inspect and len(curr_col) == 0:
                                                            if number and not property_single:
                                                                if number != '-' and number != '--':
                                                                    sub_label.append(number)
                                                        if property_single:
                                                            property_single['table_topic'] = first_col[1]
                                                            all_material.append(property_single)
                                                    if first_col[index_row] and len(fc_ns) != 0 and len(outcome_name) == 1:
                                                        material_name = outcome_name[0].replace('~', ' ')
                                                        property_single = {}
                                                        unit_search_parts.append(first_col[index_row])
                                                        for row_s in range(2, target_prop_row[prop_i] + 1):
                                                            unit_search_parts.append(
                                                                sheet.row_values(row_s)[target_prop_col[prop_i]])

                                                        number = sheet.row_values(index_row)[target_prop_col[prop_i]]
                                                        number_inspect = re.findall(self.number_pattern[prop_name],
                                                                                    str(number))
                                                        unit_search_parts.append(number)
                                                        if number_inspect:

                                                            property_single[prop_name] = number
                                                            property_single['material'] = material_name
                                                            property_single['doi'] = first_col[0]
                                                            for item in unit_search_parts:
                                                                unit_find = re.findall(self.unit_pattern[prop_name], str(item))
                                                                if unit_find:
                                                                    property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                                    K_path.append(excel_path)
                                                                    break
                                                            if 'unit' not in property_single.keys():
                                                                property_single['unit'] = 'no mentioned'
                                                            one_info = {}
                                                            for prop_index in range(
                                                                    len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = \
                                                                    sheet.row_values(target_prop_row[prop_i])[
                                                                        prop_index]
                                                                number_line_line = sheet.row_values(index_row)[
                                                                    prop_index]
                                                                one_info[prop_name_line] = number_line_line
                                                            property_single['other_info'] = one_info
                                                        elif not number_inspect and len(curr_col) != 0:
                                                            property_single['material'] = material_name
                                                            property_single['doi'] = first_col[0]
                                                            property_single[prop_name] = number
                                                            if sub_label:
                                                                property_single['child_tag'] = sub_label
                                                            for item in unit_search_parts:
                                                                unit_find = re.findall(self.unit_pattern[prop_name], str(item))
                                                                if unit_find:
                                                                    property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                                    K_path.append(excel_path)
                                                                    break
                                                            if 'unit' not in property_single.keys():
                                                                property_single['unit'] = 'no mentioned'
                                                            one_info = {}
                                                            for prop_index in range(
                                                                    len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = \
                                                                    sheet.row_values(target_prop_row[prop_i])[
                                                                        prop_index]
                                                                number_line_line = sheet.row_values(index_row)[
                                                                    prop_index]
                                                                one_info[prop_name_line] = number_line_line
                                                            property_single['other_info'] = one_info
                                                        elif not number_inspect and len(curr_col) == 0:
                                                            if number and not property_single:
                                                                sub_label.append(number)
                                                        if property_single:
                                                            property_single['table_topic'] = first_col[1]
                                                            all_material.append(property_single)
                                        else:
                                            unit_search_parts = []
                                            property_single = {}
                                            property_single['table_topic'] = first_col[1]
                                            alloy_replace = Dictionary(self.c_path).table_alloy_to_replace
                                            for alloy_model, replace in alloy_replace.items():
                                                alloy_part = re.findall(alloy_model, str(topic))
                                                for alloy in alloy_part:
                                                    find_part = re.findall(replace[0], str(alloy))
                                                    alloy_out = alloy.replace(find_part[0], replace[1])
                                                    topic = topic.replace(alloy, alloy_out)
                                            alloy_common_type = Dictionary(self.c_path).alloy_writing_type
                                            alloy_blank_type = Dictionary(self.c_path).alloy_blank_type
                                            outcome_name = []
                                            topic_tokenize = nltk.word_tokenize(topic)
                                            for word in topic_tokenize:
                                                for pattern_1 in alloy_common_type:
                                                    outcome_common = re.findall(pattern_1, str(word))
                                                    if outcome_common:
                                                        outcome_name.append(word)
                                                        break
                                            for pattern_2 in alloy_blank_type:
                                                outcome_blank = re.findall(pattern_2, str(topic))
                                                if outcome_blank and outcome_blank[0] not in outcome_name:
                                                    outcome_name.append(outcome_blank[0])
                                                    break
                                            unit_search_parts.append(first_col[3])
                                            unit_search_parts.append(topic)
                                            for row_s in range(2, 4):
                                                for prop_i in range(len(target_prop_row)):
                                                    unit_search_parts.append(sheet.row_values(row_s)[target_prop_col[prop_i]])
                                            number_search = re.findall(self.number_pattern[prop_name],
                                                                       str(sheet.col_values(0)[2]))
                                            if outcome_name and number_search:
                                                for prop_i in range(len(target_prop_row)):
                                                    property_single['material'] = outcome_name[0].replace('~', ' ')
                                                    property_single['doi'] = first_col[0]
                                                    number = sheet.row_values(3)[target_prop_col[prop_i]]
                                                    unit_search_parts.append(number)
                                                    for item in unit_search_parts:
                                                        unit_find = re.findall(self.unit_pattern[prop_name], str(item))
                                                        if unit_find:
                                                            property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                            K_path.append(excel_path)
                                                            break
                                                    if 'unit' not in property_single.keys():
                                                        property_single['unit'] = 'no mentioned'
                                                    one_info = {}
                                                    for prop_index in range(
                                                            len(sheet.row_values(target_prop_row[prop_i]))):
                                                        prop_name_line = \
                                                            sheet.row_values(target_prop_row[prop_i])[
                                                                prop_index]
                                                        number_line_line = sheet.row_values(3)[
                                                            prop_index]
                                                        one_info[prop_name_line] = number_line_line

                                                    property_single['other_info'] = one_info
                                                    property_single[prop_name] = number
                                                    all_material.append(property_single)
                                            elif not outcome_name and not number_search:
                                                for prop_i in range(len(target_prop_row)):
                                                    property_single['material'] = 'no mentioned'
                                                    # print("111")
                                                    # property_single[sheet.col_values(2)[0]] = first_col[3]

                                                    one_info = {}
                                                    for prop_index in range(
                                                            len(sheet.row_values(target_prop_row[prop_i]))):
                                                        prop_name_line = \
                                                            sheet.row_values(target_prop_row[prop_i])[
                                                                prop_index]
                                                        number_line_line = sheet.row_values(3)[
                                                            prop_index]
                                                        one_info[prop_name_line] = number_line_line

                                                    property_single['other_info'] = one_info



                                                    property_single['doi'] = first_col[0]
                                                    number = sheet.row_values(3)[target_prop_col[prop_i]]
                                                    unit_search_parts.append(number)
                                                    for item in unit_search_parts:
                                                        unit_find = re.findall(self.unit_pattern[prop_name], str(item))
                                                        if unit_find:
                                                            property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                            K_path.append(excel_path)
                                                            break
                                                    if 'unit' not in property_single.keys():
                                                        property_single['unit'] = 'no mentioned'
                                                    property_single[prop_name] = number
                                                    all_material.append(property_single)
                                            elif not outcome_name and number_search:
                                                for prop_i in range(len(target_prop_row)):
                                                    property_single['material'] = 'no mentioned'
                                                    property_single['doi'] = first_col[0]
                                                    number = sheet.row_values(3)[target_prop_col[prop_i]]
                                                    unit_search_parts.append(number)
                                                    for item in unit_search_parts:
                                                        unit_find = re.findall(self.unit_pattern, str(item))
                                                        if unit_find:
                                                            property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                            K_path.append(excel_path)
                                                            break
                                                    if 'unit' not in property_single.keys():
                                                        property_single['unit'] = 'no mentioned'
                                                    one_info = {}
                                                    for prop_index in range(
                                                            len(sheet.row_values(target_prop_row[prop_i]))):
                                                        prop_name_line = \
                                                            sheet.row_values(target_prop_row[prop_i])[
                                                                prop_index]
                                                        number_line_line = sheet.row_values(3)[
                                                            prop_index]
                                                        one_info[prop_name_line] = number_line_line

                                                    property_single['other_info'] = one_info
                                                    property_single[prop_name] = number
                                                    all_material.append(property_single)

                                elif any(word in topic.lower() for word in self.prop_pattern_words[prop_name]) and all(mat in first_col for mat in table_mats_name):
                                    # print(topic)
                                    topic_names.append(excel_path)
                                    search_outcome = []
                                    target_prop_row = []
                                    target_prop_col = []
                                    for line_index in range(2, 4, 1):
                                        search_line = sheet.row_values(line_index)[1:]
                                        unit_i = 1
                                        for unit in search_line:
                                            outcome_words = None
                                            unit_words = nltk.word_tokenize(str(unit).lower())
                                            # if any(word in str(unit).lower() for word in
                                            #        self.prop_pattern_words[prop_name]):
                                            #     all_names.append(str(unit))

                                            # if any(word in unit_words for word in self.prop_pattern_words[prop_name]) and len(unit_words)<=3:
                                            #     all_names[str(unit)] = excel_path
                                            #     target_prop_row.append(line_index)
                                            #     target_prop_col.append(unit_i)
                                            #     search_outcome.append(unit)

                                            # print(self.prop_pattern[prop_name])
                                            if any(word in str(unit).lower() for word in self.prop_pattern[prop_name]):
                                                # print(unit)
                                                all_names.append(excel_path)
                                                target_prop_row.append(line_index)
                                                target_prop_col.append(unit_i)
                                                search_outcome.append(unit)
                                            unit_i += 1
                                    if search_outcome:
                                        # print(search_outcome)
                                        first_col = sheet.col_values(0)
                                        alloy_replace = Dictionary(self.c_path).table_alloy_to_replace
                                        for alloy_model, replace in alloy_replace.items():
                                            alloy_part = re.findall(alloy_model, str(topic))
                                            for alloy in alloy_part:
                                                find_part = re.findall(replace[0], str(alloy))
                                                alloy_out = alloy.replace(find_part[0], replace[1])
                                                topic = topic.replace(alloy, alloy_out)
                                        alloy_common_type = Dictionary(self.c_path).alloy_writing_type
                                        alloy_blank_type = Dictionary(self.c_path).alloy_blank_type
                                        outcome_name = []
                                        topic_tokenize = nltk.word_tokenize(topic)
                                        for word in topic_tokenize:
                                            for pattern_1 in alloy_common_type:
                                                outcome_common = re.findall(pattern_1, str(word))
                                                if outcome_common:
                                                    outcome_name.append(word)
                                                    break
                                        for pattern_2 in alloy_blank_type:
                                            outcome_blank = re.findall(pattern_2, str(topic))
                                            if outcome_blank:
                                                outcome_name.append(outcome_blank[0])
                                                break
                                        fc_ns = []
                                        for cell in sheet.col_values(0)[1:]:
                                            fc_n = re.findall(self.number_pattern[prop_name], str(cell))
                                            alphabet_search = re.findall("[A-Za-z]", str(cell))
                                            if fc_n and not alphabet_search:
                                                fc_ns.append(cell)
                                        len_col = len(sheet.row_values(3))
                                        alloy_name_col = None
                                        alloy_name_search = []
                                        if len_col <= 3:
                                            for col_i in range(len_col):
                                                col_info = sheet.col_values(col_i)
                                                if col_i == 0:
                                                    col_info = sheet.col_values(col_i)[2:]
                                                if col_info:
                                                    for cell in col_info:
                                                        for pattern_1 in alloy_common_type:
                                                            outcome_common = re.findall(pattern_1, str(cell))
                                                            if outcome_common:
                                                                alloy_name_col = col_i
                                                                alloy_name_search.append(col_i)
                                                        for pattern_2 in alloy_blank_type:
                                                            outcome_blank = re.findall(pattern_2, str(cell))
                                                            if outcome_blank:
                                                                alloy_name_col = col_i
                                                                alloy_name_search.append(col_i)
                                        else:
                                            for col_i in range(3):
                                                col_info = sheet.col_values(col_i)
                                                if col_i == 0:
                                                    col_info = sheet.col_values(col_i)[2:]
                                                if col_info:
                                                    for cell in col_info:
                                                        for pattern_1 in alloy_common_type:
                                                            outcome_common = re.findall(pattern_1, str(cell))
                                                            if outcome_common:
                                                                alloy_name_col = col_i
                                                                alloy_name_search.append(col_i)
                                                        for pattern_2 in alloy_blank_type:
                                                            outcome_blank = re.findall(pattern_2, str(cell))
                                                            if outcome_blank:
                                                                alloy_name_col = col_i
                                                                alloy_name_search.append(col_i)
                                        if not alloy_name_search:
                                            alloy_name_col = 0
                                        else:
                                            alloy_name_col = alloy_name_search[0]
                                        if len(first_col) > 4:
                                            for prop_i in range(len(target_prop_row)):
                                                sub_label = []
                                                curr_col = []
                                                for index_row in range(target_prop_row[prop_i] + 1, len(first_col)):
                                                    unit_search_parts = []
                                                    unit_search_parts.append(topic)
                                                    if len(fc_ns) == 0:
                                                        name_col = sheet.col_values(alloy_name_col)
                                                        material_name = name_col[index_row]
                                                        property_single = {}
                                                        number = sheet.row_values(index_row)[target_prop_col[prop_i]]
                                                        number_inspect = re.findall(self.number_pattern[prop_name],
                                                                                    str(number))

                                                        unit_search_parts.append(first_col[index_row])
                                                        unit_search_parts.append(number)
                                                        for unit in sheet.row_values(target_prop_row[0]):
                                                            unit_search_parts.append(unit)
                                                        for row_s in range(2, target_prop_row[prop_i] + 1):
                                                            unit_search_parts.append(
                                                                sheet.row_values(row_s)[target_prop_col[prop_i]])
                                                        if number_inspect:
                                                            one_info = {}
                                                            for prop_index in range(
                                                                    len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = \
                                                                sheet.row_values(target_prop_row[prop_i])[
                                                                    prop_index]
                                                                number_line_line = sheet.row_values(index_row)[
                                                                    prop_index]
                                                                one_info[prop_name_line] = number_line_line
                                                            curr_col.append(number)
                                                            property_single[prop_name] = number
                                                            property_single['other_info'] = one_info
                                                            property_single['material'] = material_name
                                                            property_single['doi'] = first_col[0]
                                                            if sub_label:
                                                                property_single['child_tag'] = sub_label
                                                            for item in unit_search_parts:
                                                                unit_find = re.findall(self.unit_pattern[prop_name],
                                                                                       str(item))
                                                                if unit_find:
                                                                    property_single['unit'] = unit_find[0].replace(
                                                                        'degC', '°C')
                                                                    K_path.append(excel_path)
                                                            if 'unit' not in property_single.keys():
                                                                property_single['unit'] = 'no mentioned'
                                                        elif not number_inspect and len(curr_col) != 0:
                                                            property_single['material'] = material_name
                                                            one_info = {}
                                                            for prop_index in range(
                                                                    len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = \
                                                                    sheet.row_values(target_prop_row[prop_i])[
                                                                        prop_index]
                                                                number_line_line = sheet.row_values(index_row)[
                                                                    prop_index]
                                                                one_info[prop_name_line] = number_line_line
                                                            property_single['other_info'] = one_info
                                                            property_single['doi'] = first_col[0]
                                                            property_single[prop_name] = number
                                                            if sub_label:
                                                                property_single['child_tag'] = sub_label
                                                            for item in unit_search_parts:
                                                                unit_find = re.findall(self.unit_pattern[prop_name],
                                                                                       str(item))
                                                                if unit_find:
                                                                    property_single['unit'] = unit_find[0].replace(
                                                                        'degC', '°C')
                                                                    K_path.append(excel_path)
                                                                    break
                                                            if 'unit' not in property_single.keys():
                                                                property_single['unit'] = 'no mentioned'
                                                            one_info = {}
                                                            for prop_index in range(
                                                                    len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = \
                                                                    sheet.row_values(target_prop_row[prop_i])[
                                                                        prop_index]
                                                                number_line_line = sheet.row_values(index_row)[
                                                                    prop_index]
                                                                one_info[prop_name_line] = number_line_line
                                                        elif not number_inspect and len(curr_col) == 0:
                                                            if number and not property_single:
                                                                if number != '-' and number != '--':
                                                                    sub_label.append(number)
                                                        if property_single:
                                                            property_single['table_topic'] = first_col[1]
                                                            all_material.append(property_single)
                                                    if first_col[index_row] and len(fc_ns) != 0 and len(
                                                            outcome_name) == 1:
                                                        material_name = outcome_name[0].replace('~', ' ')
                                                        property_single = {}
                                                        unit_search_parts.append(first_col[index_row])
                                                        for row_s in range(2, target_prop_row[prop_i] + 1):
                                                            unit_search_parts.append(
                                                                sheet.row_values(row_s)[target_prop_col[prop_i]])

                                                        number = sheet.row_values(index_row)[target_prop_col[prop_i]]
                                                        number_inspect = re.findall(self.number_pattern[prop_name],
                                                                                    str(number))
                                                        unit_search_parts.append(number)
                                                        if number_inspect:
                                                            property_single[prop_name] = number
                                                            property_single['material'] = material_name
                                                            property_single['doi'] = first_col[0]
                                                            for item in unit_search_parts:
                                                                unit_find = re.findall(self.unit_pattern[prop_name],
                                                                                       str(item))
                                                                if unit_find:
                                                                    property_single['unit'] = unit_find[0].replace(
                                                                        'degC', '°C')
                                                                    K_path.append(excel_path)
                                                                    break
                                                            if 'unit' not in property_single.keys():
                                                                property_single['unit'] = 'no mentioned'
                                                            one_info = {}
                                                            for prop_index in range(
                                                                    len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = \
                                                                    sheet.row_values(target_prop_row[prop_i])[
                                                                        prop_index]
                                                                number_line_line = sheet.row_values(index_row)[
                                                                    prop_index]
                                                                one_info[prop_name_line] = number_line_line
                                                            property_single['other_info'] = 'no mentioned'
                                                        elif not number_inspect and len(curr_col) != 0:
                                                            property_single['material'] = material_name
                                                            property_single['doi'] = first_col[0]
                                                            property_single[prop_name] = number
                                                            if sub_label:
                                                                property_single['child_tag'] = sub_label
                                                            for item in unit_search_parts:
                                                                unit_find = re.findall(self.unit_pattern[prop_name],
                                                                                       str(item))
                                                                if unit_find:
                                                                    property_single['unit'] = unit_find[0].replace(
                                                                        'degC', '°C')
                                                                    K_path.append(excel_path)
                                                                    break
                                                            if 'unit' not in property_single.keys():
                                                                property_single['unit'] = 'no mentioned'
                                                            one_info = {}
                                                            for prop_index in range(
                                                                    len(sheet.row_values(target_prop_row[prop_i]))):
                                                                prop_name_line = \
                                                                    sheet.row_values(target_prop_row[prop_i])[
                                                                        prop_index]
                                                                number_line_line = sheet.row_values(index_row)[
                                                                    prop_index]
                                                                one_info[prop_name_line] = number_line_line
                                                            property_single['other_info'] = 'no mentioned'
                                                        elif not number_inspect and len(curr_col) == 0:
                                                            if number and not property_single:
                                                                sub_label.append(number)
                                                        if property_single:
                                                            property_single['table_topic'] = first_col[1]
                                                            all_material.append(property_single)
                                        else:
                                            unit_search_parts = []
                                            property_single = {}
                                            property_single['table_topic'] = first_col[1]
                                            alloy_replace = Dictionary(self.c_path).table_alloy_to_replace
                                            for alloy_model, replace in alloy_replace.items():
                                                alloy_part = re.findall(alloy_model, str(topic))
                                                for alloy in alloy_part:
                                                    find_part = re.findall(replace[0], str(alloy))
                                                    alloy_out = alloy.replace(find_part[0], replace[1])
                                                    topic = topic.replace(alloy, alloy_out)
                                            alloy_common_type = Dictionary(self.c_path).alloy_writing_type
                                            alloy_blank_type = Dictionary(self.c_path).alloy_blank_type
                                            outcome_name = []
                                            topic_tokenize = nltk.word_tokenize(topic)
                                            for word in topic_tokenize:
                                                for pattern_1 in alloy_common_type:
                                                    outcome_common = re.findall(pattern_1, str(word))
                                                    if outcome_common:
                                                        outcome_name.append(word)
                                                        break
                                            for pattern_2 in alloy_blank_type:
                                                outcome_blank = re.findall(pattern_2, str(topic))
                                                if outcome_blank and outcome_blank[0] not in outcome_name:
                                                    outcome_name.append(outcome_blank[0])
                                                    break
                                            unit_search_parts.append(first_col[3])
                                            unit_search_parts.append(topic)
                                            for row_s in range(2, 4):
                                                for prop_i in range(len(target_prop_row)):
                                                    unit_search_parts.append(
                                                        sheet.row_values(row_s)[target_prop_col[prop_i]])
                                            number_search = re.findall(self.number_pattern[prop_name],
                                                                       str(sheet.col_values(0)[2]))
                                            if outcome_name and number_search:
                                                for prop_i in range(len(target_prop_row)):
                                                    property_single['material'] = outcome_name[0].replace('~', ' ')
                                                    one_info = {}
                                                    for prop_index in range(
                                                            len(sheet.row_values(target_prop_row[prop_i]))):
                                                        prop_name_line = \
                                                            sheet.row_values(target_prop_row[prop_i])[
                                                                prop_index]
                                                        number_line_line = sheet.row_values(3)[
                                                            prop_index]
                                                        one_info[prop_name_line] = number_line_line
                                                    property_single['other_info'] = one_info
                                                    property_single['doi'] = first_col[0]
                                                    number = sheet.row_values(3)[target_prop_col[prop_i]]
                                                    unit_search_parts.append(number)
                                                    for item in unit_search_parts:
                                                        unit_find = re.findall(self.unit_pattern[prop_name], str(item))
                                                        if unit_find:
                                                            property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                            K_path.append(excel_path)
                                                            break
                                                    if 'unit' not in property_single.keys():
                                                        property_single['unit'] = 'no mentioned'
                                                    property_single[prop_name] = number
                                                    all_material.append(property_single)
                                            elif not outcome_name and not number_search:
                                                for prop_i in range(len(target_prop_row)):
                                                    property_single['material'] = 'no mentioned'
                                                    one_info = {}
                                                    for prop_index in range(
                                                            len(sheet.row_values(target_prop_row[prop_i]))):
                                                        prop_name_line = \
                                                            sheet.row_values(target_prop_row[prop_i])[
                                                                prop_index]
                                                        number_line_line = sheet.row_values(3)[
                                                            prop_index]
                                                        one_info[prop_name_line] = number_line_line
                                                    # print("222")
                                                    # property_single[sheet.col_values(2)[0]] = first_col[3]
                                                    property_single['other_info'] = one_info

                                                    property_single['doi'] = first_col[0]
                                                    number = sheet.row_values(3)[target_prop_col[prop_i]]
                                                    unit_search_parts.append(number)
                                                    for item in unit_search_parts:
                                                        unit_find = re.findall(self.unit_pattern[prop_name], str(item))
                                                        if unit_find:
                                                            property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                            K_path.append(excel_path)
                                                            break
                                                    if 'unit' not in property_single.keys():
                                                        property_single['unit'] = 'no mentioned'
                                                    property_single[prop_name] = number
                                                    all_material.append(property_single)
                                            elif not outcome_name and number_search:
                                                for prop_i in range(len(target_prop_row)):
                                                    property_single['material'] = 'no mentioned'
                                                    property_single['doi'] = first_col[0]
                                                    number = sheet.row_values(3)[target_prop_col[prop_i]]
                                                    unit_search_parts.append(number)
                                                    for item in unit_search_parts:
                                                        unit_find = re.findall(self.unit_pattern, str(item))
                                                        if unit_find:
                                                            property_single['unit'] = unit_find[0].replace('degC', '°C')
                                                            K_path.append(excel_path)
                                                            break
                                                    one_info = {}
                                                    for prop_index in range(
                                                            len(sheet.row_values(target_prop_row[prop_i]))):
                                                        prop_name_line = \
                                                            sheet.row_values(target_prop_row[prop_i])[
                                                                prop_index]
                                                        number_line_line = sheet.row_values(3)[
                                                            prop_index]
                                                        one_info[prop_name_line] = number_line_line
                                                    # print("222")
                                                    # property_single[sheet.col_values(2)[0]] = first_col[3]
                                                    property_single['other_info'] = one_info
                                                    if 'unit' not in property_single.keys():
                                                        property_single['unit'] = 'no mentioned'
                                                    property_single[prop_name] = number
                                                    all_material.append(property_single)


                        except Exception as e:
                            self.log_wp.print_log("An error in file:%s-sheet:%s---%s!", excel_path, sheet_i, e)

                    if all_material:
                        number_prop += 1
                        doi = excel_path.replace(".xlsx","")
                        doi = doi.replace("/", "-")
                        property_all[doi] = all_material
                except Exception as e:
                    self.log_wp.print_log("can't open %s ", excel_path)
                    self.log_wp.print_log("%s", str(e))
                    self.log_wp.print_log("%s", "--" * 25)
        return property_all
#
# c_path = "dictionary.ini"
# excels_path = r"C:\Users\win\Desktop\test_table"  #r"C:\Users\win\Desktop\test\pipeline\excel\all"
# out_path = r"C:\Users\win\Desktop\test\pipeline\excel\test"

# with open(r"E:\文本挖掘\工作三-工艺抽取\文章预先文稿\files\tokens.json", mode="r", encoding="utf-8") as f_t:
#     tokens = json.load(f_t)
# with open(r"E:\文本挖掘\工作三-工艺抽取\文章预先文稿\files\chunks.json", mode="r", encoding="utf-8") as f_c:
#     chunks = json.load(f_c)
# dict_all = list()
# dict_all.extend(tokens)
# dict_all.extend(chunks)

# te = TableExtraction(excels_path, c_path)
# all_composition = te.composition_triple_extraction()
# with open(r"C:\Users\win\Desktop\test\pipeline\excel\test\compositions_test.json", mode="w+", encoding="utf-8") as file:
#     json.dump(all_composition, file)

# with open(r"C:\Users\win\Desktop\test\pipeline\compositions.json", mode="r", encoding="utf-8") as file:
#     composition = json.load(file)
# te = TableExtraction(excels_path, c_path)
# extract_prop = ["solution","age"]
# all_composition_ = te.property_info_extraction(extract_prop,composition)
# with open(r"C:\Users\win\Desktop\test\pipeline\actions.json", mode="w+", encoding="utf-8") as file_:
#     json.dump(all_composition_, file_)


