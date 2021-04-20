# Attention Based LSTM for Aspect-level Sentiment Classificaion

Sentiment Analysis is an important task in Natural Language Processing and is applied in multiple areas. Sentiment analysis can help identify the sentiment behind an opinion or statement. This task works in the setting that the given text has only one aspect and polarity. However, there might be several aspects that have triggered the identified sentiment. In the paper, we study the Aspect Based Sentiment Analysis that takes into consideration the terms related to the aspects and identifies the sentiment associated with each aspect. For instance, “The appetizers are ok, but the service is slow.”, for aspect taste, the polarity is positive while for service, the polarity is negative. Thus to explore the connection between an aspect and the content of a sentence, we study the approach of using an Attention-based Long Short-Term Memory Network for aspect-level sentiment classification. The model can concentrate on different parts of a sentence when different aspects are given by using the attention-based mechanism. We will experiment on the dataset of SemEval 2014. The dataset consists of customer reviews. Each review contains a list of aspects and corresponding polarities.

## In order to run the files included in the github the following files and libraries are required:
### Files to download
Download glove.42B.300d.txt and add it to main folder in order to run code, can be downloaded here:  
https://www.kaggle.com/yutanakamura/glove42b300dtxt

### Required Libraries

numpy==1.20.1 
pytorch==1.7.1  
sklearn  
spacy  
python -m spacy download en_core_web_sm  
