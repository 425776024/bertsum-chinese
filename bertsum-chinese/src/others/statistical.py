import pandas as pd

path = '../../results/result_step_10001.csv'
result_step_10001 = pd.read_csv(path, sep='\t')


def apply_statis(x: pd.Series):
    real_idx = [int(i) for i in x['real_idx'].split(' ')]
    predict_idx = sorted([int(i) for i in x['predict_idx'].split(' ')])
    r_in_p = 0.0  # real_idx中，在predict_idx中的数量
    r_notin_p = 0.0  # real_idx中，不在predict_idx中的数量
    for ri in real_idx:
        if ri in predict_idx:
            r_in_p += 1.0
        else:
            r_notin_p += 1.0
    x['r_in_p'] = r_in_p / len(real_idx)
    x['r_notin_p'] = r_notin_p / len(real_idx)
    #     print(real_idx,predict_idx,x['r_in_p'],x['r_notin_p'])
    return x


def sent_sount_stas():
    # 句子量分布
    result_step_10001['len'] = result_step_10001['src'].apply(lambda x: len(x.split('[SEP]')))
    rsc = result_step_10001['len'].value_counts()
    return rsc


result_step_10001 = result_step_10001.apply(lambda x: apply_statis(x), axis=1)
res = sent_sount_stas()
print('句子量分布', res)

# 预测占比
succ = result_step_10001['r_in_p'].sum() / result_step_10001.shape[0]
print('预测占比:', succ)

# 出错占比
erro = result_step_10001['r_notin_p'].sum() / result_step_10001.shape[0]
print('出错占比:', erro)
