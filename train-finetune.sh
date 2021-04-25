# Fine-tuning presets

###############################################################################

# Korean parser models

# Korean_DeepBiaffParser_GloVe+PoS
python -m victornlp_dp.train \
  --finetune-model models/Korean_DeepBiaffParser_GloVe+PoS

# Korean_DeepBiaffParser_KoBERT+PoS
python -m victornlp_dp.train \
  --finetune-model models/Korean_DeepBiaffParser_KoBERT+PoS \
  --batch-size 24

# Korean_DeepBiaffParser_KorBERT
python -m victornlp_dp.train \
  --finetune-model models/Korean_DeepBiaffParser_KorBERT \
  --batch-size 24

# Korean_LeftToRightParser_GloVe+PoS
python -m victornlp_dp.train \
  --finetune-model models/Korean_LeftToRightParser_GloVe+PoS

# Korean_LeftToRightParser_KoBERT+PoS
python -m victornlp_dp.train \
  --finetune-model models/Korean_LeftToRightParser_KoBERT+PoS \
  --batch-size 24

# Korean_LeftToRightParser_KorBERT
python -m victornlp_dp.train \
  --finetune-model models/Korean_LeftToRightParser_KorBERT \
  --batch-size 24

###############################################################################

# English parser models

# Englsh_DeepBiaffParser_GloVe+PoS
python -m victornlp_dp.train \
  --finetune-model models/English_DeepBiaffParser_GloVe+PoS

# English_DeepBiaffParser_BERT+PoS
python -m victornlp_dp.train \
  --finetune-model models/English_DeepBiaffParser_BERT+PoS \
  --batch-size 24

# Englsh_LeftToRightParser_GloVe+PoS
python -m victornlp_dp.train \
  --finetune-model models/English_LeftToRight_GloVe+PoS

# English_LeftToRightParser_BERT+PoS
python -m victornlp_dp.train \
  --finetune-model models/English_LeftToRight_BERT+PoS \
  --batch-size 24