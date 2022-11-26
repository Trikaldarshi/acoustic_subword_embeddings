# Acoustic Subword Embeddings (ASE)

## Projected Embeddings
Acoustic Word Embedding of test set: [AWE test](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/Trikaldarshi/48f3df6c9e6081c1411a898766d43384/raw/c81c10e8bed6cfb45b06f87f16143040fae7f437/test_awe_config.json)

Acoustic Subword Embedding of test set: [ASE test](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/Trikaldarshi/4401a2bb68c642394b903ffecb758ce8/raw/2a863b412cd9d3315c58b3542889a1bf59a2f03e/test_sub_emb.json)

Reconstructed Acousti Word Embedidngs along with test set embeddings: [AWE Reconstructed + test set](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/Trikaldarshi/d75389490bf35fc157df91cb4b7a39ee/raw/56bb1b9d5e105907c442ed018e31f16632c3957d/reconstructed_words.json)

Note: The Reconstructed words start with "REC_".


## Instructions for running the codes:

#### Tokenizer demo
Example is provided in tokenizer_example.ipynb for computing tokens of a words.

#### Computing MFCCs and HuBERT features for creating AWEs
To compute HuBERT features, run the following command:

python load_save_hubert.py HUBERT_BASE ./data/ ./[add your output dir]/hubert_features/ .[add your librispeech data dir]/LibriSpeech/

To compute MFCC features, run the following command:

python load_save_mfcc_dd.py ./data/ ./[add your output dir]/mfcc_features_norm/ .[add your librispeech data dir]/LibriSpeech/


#### Extraction of AWEs for both MFCCs and HuBERT based CAE-RNN model and store them

To train a HuBERT based CAE-RNN model, run the following code:

python cae_rnn.py @cae_rnn_ssl.txt

To train a MFCC based CAE-RNN model, run the following code:

python cae_rnn.py @cae_rnn_mfcc.txt

After training the model either with HuBERT or MFCC features, run the following code for exracting the AWEs from the trained CAE-RNN model checkpoints (following code is for HuBERT based AWEs):

python cal_emb.py @cal_emb.txt

### Extraction of ASEs from AWEs and store them (Note: AWEs are derived using HuBERT as input features)
#### Train Factorisation model
python factorisation_train_loss.py @factorisation_train_loss.txt

After training factorisation model, extract and store the subword embedding from the following command:

python cal_sub_emb.py @cal_sub_emb.txt

#### Run evluation
After calculating subword embeddings, run the following exvaluation script to compute the average precision metrics (AP-SW and AP-RW)

python evaluation.py @evaluation.txt


