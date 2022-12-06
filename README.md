# Action_extractor
[![Website](https://raw.githubusercontent.com/MGEdata/SuperalloyDigger/master/pic_folder/b96627c3f326953fce3452fd175f718.png)](http:superalloydigger.mgedata.cn)

[![SuperalloyDigger.org](https://shields.mitmproxy.org/badge/https%3A%2F%2F-superalloydigger.mgedata.cn-green)](http:superalloydigger.mgedata.cn)

[![Website](https://badge.fury.io/py/SuperalloyDigger.svg)](https://pypi.org/project/SuperalloyDigger)
[![Supported Python versions](https://shields.mitmproxy.org/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://pypi.org/project/SuperalloyDigger)
----------------------
A semi-supervised text mining method was proposed to capture alloy synthesis and processing insights. Taking experimental procedure extraction of superalloys as a scenario, the actions sequence with the corresponding parameters in synthesis and processing corpus to reflect the detail fabrication procedure have been successfully extracted. In particular, a semi-supervised recommendation algorithm for token-level action and a multi-level bootstrapping algorithm for chunk-level actions are developed for small corpus with few annotations, which only require a small number of seeds to start the learning process. In total, a dataset with 9853 instances covering chemical compositions and actions was automatically extracted from a corpus of 16604 articles from Elsevier and other publishers.
 
This package is released under MIT License, please see the LICENSE file for details.

**Features**
----------------------
- An automated chemical composition and action extraction pipeline for superalloy.
- A semi-supervised recommendation algorithm for token-level action extraction and a multi-level bootstrapping algorithm for chunk-level actions extraction.
- Algorithm based on distance and number of entities, processing multiple relationship extraction for text without labeling samples.
- The structural and semantic relation for each action entity was infered and the parsing tree was constructed based on dependence grammar.
- An interdependency parser to extract the information contain composition and action data simultaneous.

**License**
----------------------
All source code is licensed under the MIT license.
