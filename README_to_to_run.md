# BERTSUM中文数据实验说明

基于论文`Fine-tune BERT for Extractive Summarization`的方法论&源代码，进行调整，在中文数据集中进行实验。

参考论文作者主页(含论文pdf & 源代码链接)：[http://nlp-yang.github.io/](http://nlp-yang.github.io/)



## 数据集

* 中文数据集：**LCSTS2.0** (A Large Scale Chinese Short Text Summarization Dataset)

* 来源：Intelligent Computing Research Center, Harbin Institute of Technology Shenzhen Graduate School(`哈尔滨工业大学深圳研究生院·智能计算研究中心`)

* 申请途径：[http://icrc.hitsz.edu.cn/Article/show/139.html](http://icrc.hitsz.edu.cn/Article/show/139.html)



## 预处理

###Step 1 下载原始数据

下载LCSTS2.0原始数据，下载途径。将`LCSTS2.0/DATA`目录下所有**PART_*.txt**文件放入`BertSum-master_Chinese/raw_data`

###Step 2 将原始文件转换成json文件存储

`BertSum-master_Chinese/src`目录下，运行：

```
python preprocess_LAI.py -mode format_raw -raw_path ../raw_data -save_path ../raw_data -log_file ../logs/preprocess.log
```

###Step 3 分句分词 & 分割文件 & 进一步简化格式

* 分句分词：首先按照符号['。', '！', '？']分句，若得到的句数少于2句，则用['，', '；']进一步分句

* 分割文件：训练集文件太大，分割成小文件便于后期训练。**分割后，每个文件包含不多于16000条记录**

`BertSum-master_Chinese/src`目录下，运行：

```
python preprocess_LAI.py -mode format_to_lines -raw_path ../raw_data -save_path ../json_data/LCSTS -log_file ../logs/preprocess.log
```

###Step 4 句子标注 & 训练前预处理

* 句子预处理：找出与参考摘要最接近的n句话(相似程度以ROUGE衡量)，标注为1(属于摘要)

```
python preprocess_LAI.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 2 -log_file ../logs/preprocess.log
```



## 模型训练

**提醒**：**First run**: For the first time, you should use single-GPU, so the code can download the BERT model. Change ``-visible_gpus 0,1,2  -gpu_ranks 0,1,2 -world_size 3`` to ``-visible_gpus 0  -gpu_ranks 0 -world_size 1``, after downloading, you could kill the process and rerun the code with multi-GPUs.



`BertSum-master_Chinese/src`目录下，运行下列三行代码其中之一：

**三行代码区别是参数 -encoder设置了不同值(classifier & transformer & rnn)分别代表三种不同的摘要层**

BERT+Classifier model:

```
python train_LAI.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 1 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 10000
```

BERT+Transformer model:
```
python train_LAI.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 1 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
```

BERT+RNN model:
```
python train_LAI.py -mode train -encoder rnn -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_rnn -lr 2e-3 -visible_gpus 1 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_rnn -use_interval true -warmup_steps 10000 -rnn_size 768 -dropout 0.1
```



**提醒**：如果训练过程被意外中断，可以通过以下代码从某个节点继续训练(-save_checkpoint_steps设置了定期储存模型信息)

以下代码将从第20,000步储存的模型继续训练(示例-encoder 设置为transformer，classifier & rnn同理)：

```
python train_LAI.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 1  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8 -train_from ../models/bert_transformer/model_step_20000.pt
```



## 模型评估

模型训练完毕后，`BertSum-master_Chinese/src`目录下，运行：

```
python train_LAI.py -mode test -bert_data_path ../bert_data/LCSTS -model_path MODEL_PATH -visible_gpus 1 -gpu_ranks 0 -batch_size 30000 -log_file LOG_FILE -result_path ../results/LCSTS -test_all -block_trigram False -test_from ../models/bert_transformer/model_step_30000.pt
```

- `MODEL_PATH` 是储存checkpoints的目录
- `RESULT_PATH` is where you want to put decoded summaries (default `../results/LCSTS`)



## 生成Oracle摘要

Oracle摘要：使用贪婪算法，在原文中找到与参考摘要最相近n句话(原代码设置n=3，可自行调整)



摘要大小调整方法：

目录`BertSum-master_Chinese/src/prepro/`：

data_builder_LAI.py: line204 - oracle_ids = greedy_selection(source, tgt, **3**)



`BertSum-master_Chinese/src`目录下，运行：

```
python train_LAI.py -mode oracle -bert_data_path ../bert_data/LCSTS -visible_gpus -1 -batch_size 30000 -log_file LCSTS_oracle -result_path ../results/LCSTS_oracle -block_trigram false
```



## 新数据集训练

如果要在新数据集上使用BERTSUM，只需：

* 原始数据格式整理成`BertSum-master_Chinese/raw_data/LCSTS_test.json`文件中数据对应格式
* 相应文件名／路径名也要做调整如：`-bert_data_path ../bert_data/LCSTS` `-log_file LCSTS_oracle` (LCSTS改成对应名称)
* 调整完后，预处理部分从**Step 3**开始即可