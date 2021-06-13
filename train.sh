# Training presets

###############################################################################

# Korean parser models: KMDP

# Korean_DeepBiaffParser_KorBERT
python -m victornlp_dp.train \
  --title KMDP_DeepBiaffParser_KorBERT \
  --language korean \
  --model deep-biaff-parser \
  -e glove-morph-kor -e pos-morph-kor \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_DeepBiaffParser_KorBERT
python -m victornlp_dp.train \
  --title KMDP_DeepBiaffParser_KorBERT \
  --language korean \
  --model deep-biaff-parser \
  -e etri-korbert \
  --loss-fn local-xbce \
  --parse-fn mst

# Korean_LeftToRightParser_GloVe+PoS
python -m victornlp_dp.train \
  --title KMDP_LeftToRightParser_GloVe+PoS \
  --language korean \
  --model left-to-right-parser \
  -e glove-morph-kor -e pos-morph-kor \
  --loss-fn local-lh \
  --parse-fn mst
  
# Korean_LeftToRightParser_KorBERT
python -m victornlp_dp.train \
  --title KMDP_LeftToRightParser_KorBERT \
  --language korean \
  --model left-to-right-parser \
  -e etri-korbert \
  --loss-fn local-lh \
  --parse-fn mst