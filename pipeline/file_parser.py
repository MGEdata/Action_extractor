import re
from chemdataextractor import Document
import os
import json

# 从html文件中解析得到自然段heading和内容，存放到json文件中

class file_parser():
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = dict()
    def filtered_ele_1(self, ele):
        text = ele.text
        refs = re.findall(r"\[\d+[^\[]*\]", text)
        for ref in refs:
            text = text.replace(ref, '')
        filter_text = text.replace("\n", '')
        return filter_text

    def filter_content(self, all_content):
        new_content = dict()
        if len(all_content.keys()) > 1:
            if "other_contents" in all_content.keys():
                all_content.pop("other_contents")
            for head, content in all_content.items():
                if content:
                    new_content[head] = content

        elif len(all_content.keys()) == 1:
            if "other_contents" in all_content.keys():
                for head, content in all_content.items():
                    a = list()
                    n_c = " ".join(content)
                    a.append(n_c)
                    new_content[head] = a
            else:
                for head, content in all_content.items():
                    if content:
                        new_content[head] = content
        return new_content

    def no_parags_label(self, doc, length):
        # 得到的是当文档中没有段落标签时，文章的内容
        parags = list()
        result = dict()
        paragraphs = doc.paragraphs
        for parag in paragraphs:
            text = parag.text
            if text[0].isupper() and len(text) > length:
                refs = re.findall(r"\[\d+[^\[]*\]", text)
                for ref in refs:
                    text = text.replace(ref, '')
                text = text.replace("\n", '')
                parags.append(text)
            else:
                refs = re.findall(r"\[\d+[^\[]*\]", text)
                for ref in refs:
                    text = text.replace(ref, '')
                text = text.replace("\n", '')
                parags[-1] += text
        result["None Heading"] = parags
        return result

    def html_run(self, path):
        all_content = dict()
        file = open(path, 'rb')
        doc = Document.from_file(file)
        file.close()
        elements = doc.elements
        first_heading = "None Heading"
        for ele in elements:
            ele_type_str = str(type(ele))
            ele_type = re.findall("chemdataextractor\.doc\.text\.([A-Za-z]+)", ele_type_str)
            if ele_type and ele_type[0] == "Paragraph":
                if first_heading not in all_content.keys():
                    all_content[first_heading] = list()
                    text = self.filtered_ele_1(ele)
                    if text:
                        all_content[first_heading].append(text)
                else:
                    text = self.filtered_ele_1(ele)
                    if text:
                        all_content[first_heading].append(text)
            elif ele_type and ele_type[0] == "Heading":
                first_heading = ele.text
        if not all_content:
            all_content = self.no_parags_label(doc, 100)
        return all_content

    def xml_run(self,path):
        all_content = dict()
        all_none_id = list()
        parag_p = ['^[Pp][A-Za-z]*(\d*\.?\d+)$', 'd1e[A-Za-z]*(\d*\.?\d+)$', '^[Ss][Ee][Cc](\d\S*)$',
                   '^[Ss][a-z]*(\d*\.?\d+)$',
                   'para(\.?\d+)']
        all_content["abstract"] = list()
        all_content["other_contents"] = list()
        file = open(path, 'rb')
        doc = Document.from_file(file)
        file.close()
        elements = doc.paragraphs
        first_heading = "None Heading"
        abs_id = None
        sec_id = None
        parag_id = None
        for ele in elements:
            if "abs" in str(ele.id) or "Abs" in str(ele.id):
                if abs_id == None:
                    abs_id = ele.id
                    all_content["abstract"].append(ele.text)
                elif abs_id and ele.id == abs_id:
                    all_content["abstract"][-1] += " "
                    all_content["abstract"][-1] += ele.text
                elif abs_id and ele.id != abs_id:
                    all_content["abstract"].append(ele.text)
                    abs_id = ele.id

            elif "sec" in str(ele.id) or "SEC" in str(ele.id):
                # all_sec.append(str(ele.text))
                if sec_id == None:
                    sec_id = ele.id
                    sec_topic = ele.text
                    all_content[sec_topic] = list()
                if sec_id and ele.id != sec_id:
                    sec_id = ele.id
                    sec_topic = ele.text
                    all_content[sec_topic] = list()
            else:
                if not (ele.text).startswith("http"):
                    all_content["other_contents"].append(ele.text)
            if sec_id:
                for pattern in parag_p:
                    parag_f = re.findall(pattern, str(ele.id))
                    if parag_f and not parag_id and sec_id:
                        parag_id = ele.id
                        all_content[sec_topic].append(ele.text)
                        break
                    if parag_f and parag_id and sec_id:
                        if parag_id == ele.id:
                            if all_content[sec_topic]:
                                all_content[sec_topic][-1] += " "
                                all_content[sec_topic][-1] += ele.text
                            else:
                                all_content[sec_topic].append(ele.text)
                            break
                        if parag_id != ele.id:
                            all_content[sec_topic].append(ele.text)
                            parag_id = ele.id
                            break
            if not sec_id:
                for pattern in parag_p:
                    parag_f = re.findall(pattern, str(ele.id))
                    if parag_f and not parag_id:
                        parag_id = ele.id
                        all_content[first_heading] = list()
                        all_content[first_heading].append(ele.text)
                        break
                    if parag_f and parag_id:
                        if parag_id == ele.id:
                            all_content[first_heading][-1] += " "
                            all_content[first_heading][-1] += ele.text
                            break
                        if parag_id != ele.id:
                            all_content[first_heading].append(ele.text)
                            parag_id = ele.id
                            break
        all_contents = self.filter_content(all_content)
        value_e = None
        for k, v in all_content.items():
            if v:
                value_e = True
        if not value_e:
            for par in elements:
                if str(par.id) not in all_none_id and str(par.id) != "None":
                    id_dict = dict()
                    all_none_id.append(id_dict)
        return all_contents

    def parse(self):
        files = os.listdir(self.file_path)
        for file in files:
            file_path_ = os.path.join(self.file_path, file)
            if file.endswith(".xml"):
                self.content[file] = self.xml_run(file_path_)
            elif file.endswith(".html"):
                self.content[file] = self.html_run(file_path_)
            else:
                print("%s is not xls or html format"% (file))
        return self.content

