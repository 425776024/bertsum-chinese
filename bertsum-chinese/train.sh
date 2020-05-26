export PATH="$PATH:/appcom/apps/chengmengli704/NLP_env/bin"

python3 src/train_LAI.py \
        -bert_base_chinese /appcom/apps/chengmengli704/pretrained_model/bert_base/bert-base-chinese/ \
        -dataset gonggao