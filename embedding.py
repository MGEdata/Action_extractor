
import regex
from gensim.models import word2vec
import logging
from chemdataextractor.doc import Paragraph
from tqdm import tqdm


NR_BASIC = regex.compile(r"^[+-]?\d*\.?\d+\(?\d*\)?+$", regex.DOTALL)
NR_AND_UNIT = regex.compile(r"^([+-]?\d*\.?\d+\(?\d*\)?+)([\p{script=Latin}|Ω|μ]+.*)", regex.DOTALL)
SPLIT_UNITS = ["K", "h", "V", "wt", "wt.", "MHz", "kHz", "GHz", "Hz", "days", "weeks",
                   "hours", "minutes", "seconds", "T", "MPa", "GPa", "at.", "mol.",
                   "at", "m", "N", "s-1", "vol.", "vol", "eV", "A", "atm", "bar",
                   "kOe", "Oe", "h.", "mWcm−2", "keV", "MeV", "meV", "day", "week", "hour",
                   "minute", "month", "months", "year", "cycles", "years", "fs", "ns",
                   "ps", "rpm", "g", "mg", "mAcm−2", "mA", "mK", "mT", "s-1", "dB",
                   "Ag-1", "mAg-1", "mAg−1", "mAg", "mAh", "mAhg−1", "m-2", "mJ", "kJ",
                   "m2g−1", "THz", "KHz", "kJmol−1", "Torr", "gL-1", "Vcm−1", "mVs−1",
                   "J", "GJ", "mTorr", "bar", "cm2", "mbar", "kbar", "mmol", "mol", "molL−1",
                   "MΩ", "Ω", "kΩ", "mΩ", "mgL−1", "moldm−3", "m2", "m3", "cm-1", "cm",
                   "Scm−1", "Acm−1", "eV−1cm−2", "cm-2", "sccm", "cm−2eV−1", "cm−3eV−1",
                   "kA", "s−1", "emu", "L", "cmHz1", "gmol−1", "kVcm−1", "MPam1",
                   "cm2V−1s−1", "Acm−2", "cm−2s−1", "MV", "ionscm−2", "Jcm−2", "ncm−2",
                   "Jcm−2", "Wcm−2", "GWcm−2", "Acm−2K−2", "gcm−3", "cm3g−1", "mgl−1",
                   "mgml−1", "mgcm−2", "mΩcm", "cm−2s−1", "cm−2", "ions", "moll−1",
                   "nmol", "psi", "mol·L−1", "Jkg−1K−1", "km", "Wm−2", "mass", "mmHg",
                   "mmmin−1", "GeV", "m−2", "m−2s−1", "Kmin−1", "gL−1", "ng", "hr", "w",
                   "mN", "kN", "Mrad", "rad", "arcsec", "Ag−1", "dpa", "cdm−2",
                   "cd", "mcd", "mHz", "m−3", "ppm", "phr", "mL", "ML", "mlmin−1", "MWm−2",
                   "Wm−1K−1", "Wm−1K−1", "kWh", "Wkg−1", "Jm−3", "m-3", "gl−1", "A−1",
                   "Ks−1", "mgdm−3", "mms−1", "ks", "appm", "ºC", "HV", "kDa", "Da", "kG",
                   "kGy", "MGy", "Gy", "mGy", "Gbps", "μB", "μL", "μF", "nF", "pF", "mF",
                   "A", "Å", "A˚", "μgL−1",'K min − 1', 'A/cm2', 'r/min', 'mg/min', 'mm/s',
                   'g/l', 'mJ/m2', 'm/min', 'mm/rev', 'mg/cm2', '°C/min', 'mL/min', 'm/s',
                   'mm/min', '°C·min−1', 's−1', 'min−1', 'h−1', 'mmh−1', '−1', 'mL', 'vol%%',
                   'L', 'vol.%%', 'ml', 'V', 'eV', 'Mpa', 'MPa', 'meV', 'mA', 'GPa', 'Gpa',
                   'keV', 'W', 'Pa', 'kW', 'F', 'nA', 'kV', 'nA', 'kPa', 'KHz', 'HV', 'Hz',
                   'Kv', 'mbar', 'kJ', 'pJ', 'mJ', 'J', 'μm', 'cm', 'nm', 'mm', 'm', 'h',
                   'days', 's−1', 's−1', 'ms', 's', 'min', 'μs', 'ns', 'weeks', 'hours',
                   'minutes', 'minute', 'hour', 'ks', 'g', 'g/L', 'pct', 'kg', 'mg', 't',
                   '°C', 'deg', 'K', '℃', '∘C', 'C', '°', 'mN', 'kN', 'N', 'mol', 'cycles']

def process(sents):

    def is_number(s):
        """Determines if the supplied string is number.

        Args:
            s: The input string.

        Returns:
            True if the supplied string is a number (both . and , are acceptable), False otherwise.
        """
        return NR_BASIC.match(s.replace(",", "")) is not None

    # 小写处理
    # 数字替换
    cde_t = Paragraph(sents)
    tokens = cde_t.tokens
    toks = []
    processed = []
    for sentence in tokens:
        for tok in sentence:
            toks.append([])
            nr_unit = NR_AND_UNIT.match(tok.text)
            if nr_unit is not None and nr_unit.group(2) in SPLIT_UNITS:
                print(nr_unit)
                # Splitting the unit from number, e.g. "5V" -> ["5", "V"].

                toks[-1] += [nr_unit.group(1), nr_unit.group(2)]
            else:
                toks[-1] += [tok]
    for i, tok in enumerate(toks):
        if isinstance(tok[0],str):
            f_text = tok[0]
        else:
            f_text = tok[0].text
        if is_number(f_text):  # Number.
        # Replace all numbers with <nUm>, except if it is a crystal direction (e.g. "(111)").
            try:
                if tokens[i - 1] == "(" and tokens[i + 1] == ")" \
                        or tokens[i - 1] == "〈" and tokens[i + 1] == "〉":
                    tok = f_text
                else:
                    tok = "<nUm>"
            except IndexError:
                tok = "<nUm>"
        else:
            tok = f_text
        processed.append(tok)
    new_sent = " ".join(processed)
    return new_sent



with open(r"./ori.txt", "r",encoding="utf-8",) as ori_file:
    all_paragraphs = ori_file.read()
paras = Paragraph(all_paragraphs)
sents = paras.sentences
new_sents = ""
for sent in tqdm(sents):
    sent_ = process(sent)
    new_sents += sent_
    new_sents += "\n"
with open(r"./reorg.txt", "w+",encoding="utf-8",) as re_file:
    re_file.write(new_sents)

sentences = word2vec.Text8Corpus(r"./reorg.txt") #经过句子开头字母小写处理的语料
model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=5)
model.save(r'.\ft_3.bin')

import fasttext
from sklearn.metrics.pairwise import cosine_similarity

# model的训练代码
text_path = r".\ft_2.bin"
model = fasttext.train_unsupervised(input=r"./reorg.txt",model="skipgram", lr=0.05, dim=100,epoch=5, minCount=5, minn=3, maxn=8, wordNgrams=4,loss="ns")
model.save_model(text_path)

# model的调用代码
# model = fasttext.load_model(r"E:\fasttext\model\ft_1.bin")
# wv_1 = model.get_word_vector("sample").reshape(1,-1)
# wv_2 = model.get_word_vector("specimen").reshape(1,-1)
# sim = cosine_similarity(wv_1, wv_2)
