# BERTSUM中文摘要

- 1.准备好json_data/ 下的那种样式的数据
- 2.运行preprocess_LAI.py把json数据转成pt形式的二进制数据
> 注意里面需要设置你自己的bert-base-chinese
> 
> 如果json数据转换失败，为空[]之类，请debug `src/prepro/http://data_builder_lai.py/ ` 102-105行代码，从那里开始处理json数据
- 3.运行src/train_LAI.py 开始训练
> src/args_config.py 下指定好你的参数和bert-base-chinese依赖
>

# 大致原理
-（不懂的加QQ/微信，** 小白，连Python、Pytorch都没入门的不要加了。🙏 浪费大家时间，先学基础 **）
- 对文本句子进行分句（0/1），1是关键句，即，关键句分类。PS：文本长度超过512，切成多段低于512的。
- 文本有7句话：则是对7个维度的[CLS]位置向量输出0/1预测
- 抽取式摘要，效果咋不好说，有好有坏，还不错

## 数据没有？
可以百度的接口生成一些训练数据，是抽取式摘要的，可以免费调用50w次
参考里面的“新闻摘要”：
https://cloud.baidu.com/product/nlp

```python

# -*- coding: utf-8 -*-

from aip import AipNlp

# 去注册生成你的
APP_ID = '22222'
API_KEY = 'xxxx'
SECRET_KEY = 'xxxxx'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

content = "3月6日，自治区政府印发划转部分国有资本充实社保基金实施方案的通知。当前，在推动国有企业深化改革的同时，通过划转部分国有资本充实社保基金，使人民群众共享国有企业发展成果，增进民生福祉，促进改革和完善基本养老保险制度，实现代际公平，增强制度的可持续性。划转范围。为我区国有及国有控股大中型企业、金融机构纳入划转范围。公益类企业、文化企业以及国务院另有规定的除外。划转对象。一是由自治区国资委监管或直接持有纳入划转范围的国有股权。二是由自治区有关部门（单位）监管或直接持有纳入划转范围的国有股权。三是由市、县（区）人民政府直接持有纳入划转范围的国有股权。划转对象涉及多个国有股东的，按照不重复划转原则进行划转。中央和地方混合持股的企业，按照第一大股东产权归属关系进行划转。划转比例。划转比例统一为纳入划转范围企业国有股权的10%。以后根据中央政策规定和我区基本养老保险基金缺口适时调整。划转基准日。本次国有股权划转原则上以2019年12月31日作为划转基准日。后续如有符合划转条件的企业，以上一年度末作为划转基准日。承接主体。我区划转的企业国有股权，委托自治区财政厅履行出资人职责的企业作为全区唯一承接主体，负责集中统一持有、专户管理和独立运营。各市、县（区）不再设立承接主体。国有资产直接划拨等制度性安排，社保基金的力量不断壮大，为我国现行养老制度的存续提供了充分安全可靠的后盾和保障。在这个过程里，国有资产的划入起到了至关重要的支柱性作用，而这也是国有资产社会使命的充分落实。"

maxSummaryLen = 300

res = client.newsSummary(content, maxSummaryLen)
print(res['summary'])

# options = {}
# options["title"] = "标题"
# client.newsSummary(content, maxSummaryLen, options)

```
