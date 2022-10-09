# Rating-Prediction-DL

All the deep learning models are implemented using [jittor](https://github.com/Jittor/jittor).

## Introduction

I have implemented LSTM, CNN, and BERT models, and ensemble learning.

All models use the embedding layer of the pretrained bert provided by jittor to encode the words of the comment text, token_dim = 768

## Result

|model|Accracy|F1|MSE|RMSE|MAE|
|---|---|---|---|---|---|
|LSTM|0.5868|0.1479|1.777|1.333|0.7440|
|CNN_BATCH_64_FEATURE_256|0.662|0.4369|0.8730|0.9343|0.4746|
|CNN_BATCH_32_FEATURE_256|0.653|0.4465|0.8595|0.9271|0.4802|
|CNN_BATCH_2_FEATURE_256|0.6447|0.4657|0.8365|0.9146|0.4801|
|CNN_BATCH_2_FEATURE_128|0.647|0.4654|0.8600|0.9274|0.4836|
|CNN_BATCH_2_FEATURE_64|0.6488|0.4566|0.8716|0.9336|0.4855|
|CNN_BATCH_2_FEATURE_256_ENSEMBLE|0.6301|0.317|1.111|1.054|0.559|
|BERT-base|**0.6976**| **0.5339**  |**0.5709**|**0.7556**|**0.3743**|

Overall, the bert model with the most parameters and using attentions performed best on this dataset and outperformed the other models in various metrics.
