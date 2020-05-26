import re
import argparse


def doc_split(doc: str):
    doc = filter(doc)
    # 给主体文本切成单个句子
    doc_sents = re.split(r"([。|\？|!|；|;])", doc)
    # 过滤空句子
    doc_sents = [str(ds) for ds in doc_sents if ds != '']
    doc_sents.append("")
    doc_sents = ["".join(i) for i in zip(doc_sents[0::2], doc_sents[1::2])]
    doc_sents = [di for di in doc_sents if len(di) >= 2]
    return doc_sents


def sent_token_split(doc):
    doc = str(doc)
    doc_split = list(doc)
    return doc_split


def filter_chinese_space(text: str) -> int:
    '''
    只给中文中的空格去除
    :param x:
    :return:
    '''
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i, new_i)
    return text


def filter(x: str):
    x = str(x).replace('<br>', '。')
    x = filter_chinese_space(x)
    dr = re.compile(r'<[^>]+>', re.S)
    dr2 = re.compile(r'{[^}]+}', re.S)
    if x is None or str(x) == 'Nan' or str(x) == 'nan':
        return x
    x = dr.sub('', x)
    x = dr2.sub('', x)
    x = x.replace('\u3000', '')
    # x = x.replace(' ', '')
    x = x.strip()
    return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def int_arr_to_str(arr: list):
    arr = [str(i) for i in arr]
    return ' '.join(arr)


def label_to_idx(label_arr: list):
    # 词袋形 label arr，转成 索引位置：[1,0,1,1,0]>>>>>[0,2,3]
    return [i for i, li in enumerate(label_arr) if li == 1]
