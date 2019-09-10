# Multi-Label Text Classification of Text Document or News

## DATASET
Toxic Comment Classification dataset. a multi-label text classfication data consisting of many wikipedia comments which have been labeled by humans according to their relative toxicity comments labels such as "toxic", "severe_toxic", "obscene", "threat", "insult", and "identity_hate". The dataset has approximately ~160k observation in total, ~125k with zero labels (toxicity) of any type, and approximately ~35k classified in one or more toxicity categories.

* Dataset Link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge


### DATA CHARACTERISTICS (TRAIN DATASET)
* Number of data points 159571
* Number data points of type toxic 15294
* Number data points of type severe_toxic 1595
* Number data points of type obscene 8449
* Number data points of type threat 478
* Number data points of type insult 7877
* Number data points of type identity_hate 1405
* Observations in one or more class 35098
* Unclassified observation 124473

### HANDLING IMBALANCED DATA
 Due to high availability of unclassified observation in the datataset. Therefore, we used the 16225 samples that are classified in atleast one sample which were around 35098 to train and validate our model. The new data (train) characteritics
 * Number of data points (records or samples) 16225
 * Number data points of type toxic 15294
 * Number data points of type severe_tocic 1595
 * Number data points of type obscene 8449
 * Number data points of type threat 478
 * Number data points of type insult 7877
 * Number data points of type identity_hate 1405
 * Observations in one or more class 35098
 * Unclassified observation 0
 * https://www.datacamp.com/community/tutorials/diving-deep-imbalanced-data 

### DATA PREPARATION
 We clean the Train and Test data, and removed punctuations
 
 
### DATA PREPROCESSING
Deep Neural Networks input layers make use of input variables to feed the network for training the model. But in this task (experiment), we're dealing with words text. How do we represent these words in order to feed our model?

In our experiment, we used densed representation of those text (comments) and their semanticity together. The advantage of using this approach is the best way for fitting neural networks onto a text data (as in our case), as well as less memory usage compared to other sparse representation approaches.

### Train, Validation, Test Splits

#### Train Data:
The sample data used to fit the model.

#### Validation Data:
The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters.

#### Test Data:
The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset

The test set is generally what is used to evaluate competing models (For example on many Kaggle competitions, the validation set is released initially along with the training set and the actual test set is only released when the competition is about to close, and it is the result of the the model on the Test set that decides the winner).

### Our Experiment Splits Ratio
Split ratio on most of the experimental datasets depends mainly on two (2) things: The number of samples in the data, and the actual model you are training.

It's always good to splits in the raio of (60%:20%:20%) in terms of (Train:Validation:Test) when you have 1 dataset. But thanks to kaggle competition spits of Train dataset and Test dataset. Now, we need to compute only the validation accuracy.

In our experiment, we used the train dataset as our (train and validation data), while the test dataset as our test data.
For the Train data, we used used cross validation to split the train data into random train and validation subset. Our model is then iteratively trained and validated on these different sets.

The Train dataset consists of (159571 samples), so when to split it into train and validaton. we have train split (0.8 of train dataset) = 127657 samples, while the validation split(0.2 of Train dataset) = 31914 samples.

Lastly, the test dataset consists of 63930 samples, therefore, we use all the samples as our test split.


### METHODS

#### Word Embedding
Two ways to feed embeddings to neural networks:

* Using your own word embeddings by training (keras)
* Using pre-trained embedding (e.g Word2vec, FastText, and Glove)


#### Without pre-trained embedding
* NN
* CNN 
* RNN
* LSTM
* GRU

#### With pre-trained embedding (with Word2Vec, FastText, Glove)
* CNN 
* RNN
* LSTM
* GRU


### EVALUATION METRICS
The evaluation measures for single-label are usually different than for multi-label. Here in single-label classfication we use simple metrics such as: 

* Accuracy
* Score
* Precision
* F1 Score or Measure
* recall
* Reference: https://romisatriawahono.net/lecture/rm/survey/machine%20learning/Zhang%20-%20Multi-Label%20Learning%20Algorithms%20-%202013.pdf

In multi-label classification, a misclassification is no longer a hard wrong or right. A prediction containing a subset of the actual classes should be considered better than a prediction that contains none of them, i.e., predicting two of the three labels correctly this is better than predicting no labels at all.

Therefore, we calculates the precision, a metric for multi-label classification of how many selected items are relevant, and also calculates the recall, a metric for multi-label classification of how many relevant items are selected.

Lastly, we calculates the F score, the weighted harmonic mean of precision and recall. This is useful for multi-label classification, where input samples can be classified as sets of labels. By only using accuracy (precision) a model would achieve a perfect score by simply assigning every class to every input. In order to avoid this, a metric should penalize incorrect class assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0) computes this, as a weighted mean of the proportion of correct class assignments vs. the proportion of incorrect class assignments. With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning correct classes becomes more important, and with beta > 1 the metric is instead weighted towards penalizing incorrect class assignments.

##### Refer to: 
* https://www.cse.iitk.ac.in/users/sigml/lec/Slides/MLL.pdf
* https://users.ics.aalto.fi/jesse/talks/Multilabel-Part01.pdf
* https://stats.stackexchange.com/questions/323154/precision-vs-recall-acceptable-limits

### EVALUATION RESULTS
* Without pre-trained Embedding:
![multi-label_without_pre-trained_embedding_with the max of the evaluation matrics (per model) _table (2)](https://user-images.githubusercontent.com/27901245/59511045-eb59ba00-8ebd-11e9-881f-8c0d94d46e08.png)
*With FastText Embedding
![fastText_trained_embedding_with the max of the evaluation matrics (per model) _table (2)](https://user-images.githubusercontent.com/27901245/59510860-7c7c6100-8ebd-11e9-962f-edc4be895db4.png)
*With Glove Embedding
![Glove_trained_embedding_with the max of the evaluation matrics (per model) _table](https://user-images.githubusercontent.com/27901245/59510944-b5b4d100-8ebd-11e9-898d-f5501bcfcdaa.png)
*With Word2Vec Embedding
![Word2Vec_pre-trained_embedding_with the max of the evaluation matrics (per model) _table (2)](https://user-images.githubusercontent.com/27901245/59510803-4d65ef80-8ebd-11e9-8a6b-f0ff7c5978f6.png)

### COMPARISON OF RESULT WITH [1]
![ResultComparisonTable](https://user-images.githubusercontent.com/27901245/64582969-ff676780-d397-11e9-8c77-c289352ab4b3.png)
