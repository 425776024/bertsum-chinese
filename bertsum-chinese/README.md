# BERTSUM中文摘要抽取代码（魔改）

- 1.准备好json_data/ 下的那种样式的数据
- 2.运行src/preprocess_LAI.py把json数据转成pt形式的二进制数据
> 注意里面需要设置你自己的bert-base-chinese
- 3.运行src/train_LAI.py 开始训练
> src/args_config.py 下指定好你的参数和bert-base-chinese依赖
>
