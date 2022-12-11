import re
import nltk
from dictionary import Dictionary


class PhraseParse:
    def __init__(self, txt, C_path):
        self.C_path = C_path
        self.dict_info = Dictionary(self.C_path)
        self.txt = txt
    def alloy_sub_search(self):
        txt = self.txt
        alloy_write_type = self.dict_info.alloy_writing_type
        alloy_blank_type = self.dict_info.alloy_blank_type
        outcome_filter = {}
        word_list = nltk.word_tokenize(txt)
        word_id = {}
        s = 0
        for word in word_list:
            word_id[word] = s
            s = s + 1
        id_word = {value: key for key, value in word_id.items()}
        sub_id = {}
        sub_order = []
        for pattern in alloy_blank_type:
            outcome = re.findall(pattern, txt)
            if outcome:
                for name in outcome:
                    sub_number_id = word_id[name]
                    sub_id[name] = sub_number_id
        len_type = len(alloy_write_type)
        for word in word_list:
            for i in range(0, len_type):
                outcome_filter[i] = re.findall(alloy_write_type[i], word)
            for k in range(0, len_type):
                if outcome_filter[k]:
                    id_word = word_id[word]
                    sub_id[word] = id_word
                    break
        name_order = sorted(sub_id.items(), key=lambda x: x[1], reverse=False)
        for item in name_order:
            check = re.findall("\d",item[0])
            if check:
                sub_order.append(item[0])
        return sub_order, sub_id