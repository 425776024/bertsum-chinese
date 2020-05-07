import re, json


def filter(x: str):
    dr = re.compile(r'<[^>]+>', re.S)
    dr2 = re.compile(r'{[^>]+}', re.S)
    if x is None or x == 'nan':
        return x
    x = dr.sub('', x)
    x = dr2.sub('', x)
    x = x.replace('\u3000', '').strip()
    return x


def have_dirty_key(doc):
    dirty_key = ['function()', 'show()', 'hide()']
    for di in dirty_key:
        if di in doc:
            return True
    if str(doc) == 'nan': return True

    return False


def paser_out_label(doc_sents: list, key_sents: list):
    label_arr = []
    match_num = 0
    for si in range(len(doc_sents)):
        mac_s = doc_sents[si]
        for ks in key_sents:
            if mac_s in ks or ks in mac_s:
                label_arr.append(si)
                match_num += 1
                break
    try:
        assert match_num > 0, '一句没匹配到'
        assert match_num == len(key_sents), '关键句未匹配完全'

    except:
        for ks in key_sents:
            if have_dirty_key(ks):
                print('关键句中存在脏数据，导致未匹配到')
                break
        if match_num >= 1: return label_arr
        return None

    return label_arr


def sent_token_split(doc: str):
    doc_split = list(doc)
    return doc_split


def doc_split(doc: str):
    doc_sents = re.split('([。\?！；;])', doc)
    doc_sents = [str(ds) for ds in doc_sents if ds != '']
    doc_sents.append('')
    doc_sents = [''.join(i) for i in zip(doc_sents[0::2], doc_sents[1::2])]
    return doc_sents


def format_to_json(doc_sents_arr, idx_arr):
    token_docs = [sent_token_split(sent) for sent in doc_sents_arr]
    json_item = {'src': token_docs, 'ids': idx_arr}
    return json_item


def save_data_arr_to_json(data_arr_iter, chunk_size=2000, file_name='data/json/train'):
    dataset = []
    p_ct = 0
    for data_item_i in data_arr_iter:
        doc_sents = data_item_i['doc_sents']
        key_sents = data_item_i['key_sents']
        label_arr = paser_out_label(doc_sents, key_sents)
        if label_arr is None or len(label_arr) == 0:
            continue
        json_dict = format_to_json(doc_sents, label_arr)
        dataset.append(json_dict)
        if len(dataset) >= chunk_size:
            path = '{:s}.chunk_size_{:d}.{:d}.json'.format(file_name, chunk_size, p_ct)
            with open(path, 'w', encoding='utf-8') as save:
                tp_js = json.dumps(dataset, ensure_ascii=False)
                save.write(tp_js)
                save.write('\n')
                dataset = []
                print('saved:', path)
    if len(dataset) > 0:
        path = '{:s}.chunk_size_{:d}.{:d}.json'.format(file_name, len(dataset), p_ct)
        with open(path, 'w', encoding='utf-8') as save:
            tp_js = json.dumps(dataset, ensure_ascii=False)
            save.write(tp_js)
            save.write('\n')
            dataset = []
            print('saved:', path)
