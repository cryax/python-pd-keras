# -*- coding: utf-8 -*-
from google import search
for url in search('ntu', tld='es',num=1, lang='es'):
    print(url)