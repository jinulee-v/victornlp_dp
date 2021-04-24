# Training presets

###############################################################################

# Korean parser models

# Korean_DeepBiaffParser_GloVe+PoS
python -m victornlp_dp.train \
  --title Korean_DeepBiaffParser_GloVe+PoS \
  --language korean \
  --model deep-biaff-parser \
  -e glove-wp-kor -e pos-wp-kor \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_DeepBiaffParser_KoBERT+PoS
python -m victornlp_dp.train \
  --title Korean_DeepBiaffParser_KoBERT+PoS \
  --language korean \
  --model deep-biaff-parser \
  -e kobert -e pos-wp-kor \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_DeepBiaffParser_KorBERT
python -m victornlp_dp.train \
  --title Korean_DeepBiaffParser_KorBERT \
  --language korean \
  --model deep-biaff-parser \
  -e etri-korbert \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_LeftToRightParser_GloVe+PoS
python -m victornlp_dp.train \
  --title Korean_LeftToRightParser_GloVe+PoS \
  --language korean \
  --model left-to-right-parser \
  -e glove-wp-kor -e pos-wp-kor \
  --loss-fn local-lh \
  --parse-fn beam

# Korean_LeftToRightParser_KoBERT+PoS
python -m victornlp_dp.train \
  --title Korean_LeftToRightParser_KoBERT+PoS \
  --language korean \
  --model left-to-right-parser \
  -e kobert -e pos-wp-kor \
  --loss-fn local-lh \
  --parse-fn beam

# Korean_LeftToRightParser_GloVe+PoS
python -m victornlp_dp.train \
  --title Korean_LeftToRightParser_KorBERT \
  --language korean \
  --model left-to-right-parser \
  -e etri-korbert \
  --loss-fn local-lh \
  --parse-fn beam

###############################################################################

# English parser models

# Englsh_DeepBiaffParser_GloVe+PoS
python -m victornlp_dp.train \
  --title English_DeepBiaffParser_GloVe+PoS \
  --language english \
  --model deep-biaff-parser \
  -e glove-eng -e pos-eng \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_DeepBiaffParser_KoBERT+PoS
python -m victornlp_dp.train \
  --title English_DeepBiaffParser_BERT+PoS \
  --language english \
  --model deep-biaff-parser \
  -e bert-base-uncased -e pos-eng \
  --loss-fn local-xbce \
  --parse-fn mst

# Englsh_LeftToRightParser_GloVe+PoS
python -m victornlp_dp.train \
  --title English_LeftToRight_GloVe+PoS \
  --language english \
  --model left-to-right-parser \
  -e glove-eng -e pos-eng \
  --loss-fn local-lh \
  --parse-fn beam

# Korean_LeftToRightParser_KoBERT+PoS
python -m victornlp_dp.train \
  --title English_LeftToRight_BERT+PoS \
  --language english \
  --model left-to-right-parser \
  -e bert-base-uncased -e pos-eng \
  --loss-fn local-lh \
  --parse-fn beam