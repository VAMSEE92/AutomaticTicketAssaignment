# Automatic Ticket Assaignment

### Problem Statement
An incident ticket is created by various groups of people within the organization to resolve an issue as quickly as
possible based on its severity. Whenever an incident is created, it reaches the Service desk team and then it gets
assigned to the respective teams to work on the incident. The Service Desk team will perform basic analysis on the
user's requirement, identify the issue based on given descriptions and assign it to the respective teams.

![incident Image](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-002.jpg)

Below are the challenges of the manual assaignment of these incidents
##### -> More resource usage and expenses
##### -> Human errors - Incidents get assigned to the wrong assignment groups
##### -> Delay in assigning the tickets
##### -> More resolution times
##### -> If a particular ticket takes more time in analysis, other productive tasks get affected for the Service Desk

### Objective
The objective of the project is to build an AI-based classifier model to assign the tickets to right functional
groups using:
##### -> Different classification models.
##### -> Transfer learning to use pre-built models
##### -> Set the optimizers, loss functions, epochs, learning rate, batch size, check pointing and early stopping to achieve an accuracy of at least 85% 

### Machine Learning Process

![ML image](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-003.jpg)
##### 
### Steps performed 
1) Reading and Merging the dataset
2) Data Preprocessing
3) Model Building 

### Reading and Merging the dataset
In this step we visualized patterns and text features in the data
### Data Preprocessing
In data Preprocessing we performed the below steps
##### -> Language detection
##### -> Language translation
##### -> Text preprocessing
##### -> Visualizing processed text with and without removing stop words

### Model Building
For model building, the accuracy has been calculated both on traditional ML classification algorithm and deep
learning algorithm using LSTM. Machine Learning algorithms like Decision tree, Random Forest, Naïve Bayes,
KNN and Logistic Regression have been compared.
#### ML Model performances
![ML performances](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-033.png) 
##### 
#### Conclusion
##### -> None of the models could reach the accuracy of 60 % in test even with hyper parameters.
##### -> Seeing the huge difference in training and test accuracy, it can be concluded that the models are over fitting
##### -> Precision is the ratio of true positive and sum of true positive and false positive. Hence the lower the value of precision in the result indicates, higher will be the false positive in the model prediction
##### -> Recall is the ratio of true positive and sum of true positive and false negative. Hence lower the value of recall indicates higher false negatives in the model prediction.
##### -> Ideally higher the f1 score, the better the model performs but the data indicated the other way round
### LSTM Model
We performed two LSTM models with and without glove embedding
#### Conclusion
LSTM with single layer gives maximum of 56% accuracy in test while LSTM with glove improves it to 62% but
the model still is overfit.

### Pre-Trained Models
#### Observations from Pre-Trained Models
1. The dataset have been tested on 3 Pretrained Models namely ULMFiT, Fasttext and BERT
2. With ULMFiT, we achieved maximum training accuracy as 77% and test accuracy as 67%.
3. With BERT, the maximum training accuracy achieved is 55 and test accuracy as 46%
4. With Fasttext, the maximum training accuracy achieved is 91% and test accuracy as 66%
##### 
![Pre-trained models](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-047.png)
### Bidirectional LSTM Performance with Top 5 Groups

![LSTMBi Model](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-048.png)
![LSTMBi acc](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-049.png)
#### Graph
![LSTMBi graph](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-050.png)
#### classification Repost
![LSTMBi cr](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-051.png)
#### Prediction
For Prediction , randomly few texts have been picked from each group and the same have been tested
against the model prediction
##### 
![LSTMBi cr](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-052.png)
#### Conclusion
1. From the classification report it can be seen that the group 9 is having least f1 score, and precision.
Hence the model is predicting GRP 9 text wrongly as GRP_0
2. The maximum accuracy that can be achieved 93% in training and 90 % in test with slight overfitting.
### ULMFit Performance
![ulmfit](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-054.png)
#### Conclusion
Just like BiLSTM model, the classification report of ULMFit states group 9 is having least f1 scor
e, and precision. Hence the model is predicting GRP 9 text wrongly as GRP_0
### Fasttext Performance 
![Fasttext](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-058.png)
#### Conclusion
From the classification report it can be seen that the group 9 is having least f1 score, and precisi
on. Hence the model is predicting GRP 9 text wrongly as GRP_0
### BERT Performance
![BERT](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-061.png)
#### Conclusion
1. From the above implementation of the BERT model we see that performance on top 5 groups of
the data-set is close to 96.5% in training and 91.5% in validation & test.
2. We see very less over-fitting this time while using the BERT model.
3. We see prediction of individual groups from random samples that we have picked, has been
done correctly. Only prediction related to GROUP 9 is not correct which is consistent with other
models.
4. In the BERT model we have used pre-trained "BERT-base-uncased" which contains 110M
parameters. We can also try “BERT -large-uncased" pre-trained model and see the
performance. Due to lack of hardware that would be necessary to run “BERT -large-uncased"
pre-trained model, we did try it.
#### Final Performance Report
![finalePer](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-063.png)
### Deployment
1. There are different ways of deploying the model
#####   -> Using Flask as web application
#####   -> Using Heroku
2. The same can also be done from google collab using flask but one cannot see the actual web page, we can only retrieve the results from the hosted environment.
3. There are different IDE for model execution. When we want to run line by line and check the output, it’s idle to go with Jupyter notebooks. When we want to run a chunk of code altogether, Spyder is the better choice and if you want your whole project to look organized, has a lot of files and want to make it look in a structured way, it’s good to go with PyCharm IDE.
4. We have used Flask to deploy the model as web Service.
5. There are three steps to follow while deploying model using Flask:
##### -> Loading of the saved model(either using pickle or load_method of tensor flow)
##### -> Redirecting the API to the home page index.html
##### -> Redirecting the API to predict the result(Assignment group)
6. Below is the folder structure that is used for deployment
![folderStr](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-064.png)
##### 
7. Web Page view of the bi-LSTM model index.html page
![home Webpage](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-065.png)
##### 
8. Web Page view of the bi-LSTM model result page
![home Webpage](https://github.com/VAMSEE92/AutomaticTicketAssaignment/blob/main/Images/image-066.png)
##### 
### Limitations
1. From the preprocessing, it is found that the data is highly imbalanced and contains non-English Text.
2. Google translation was not able to convert all non-English text to English language and hence there
remains some discrepancy in the dataset.

3. Due to data being imbalanced, even after trying different ML models, deep learning models, pretrained
models, different embeddings, none of the models could achieve a minimum accuracy of 70% in test
data. Hence the model performance was tested on top 5 frequent groups.

4. While implementing ULMFit, when the language and classification model was created using
TextLMDataBunch and TextClassDataBunch then the model was giving very low accuracy within range of
0-10%.

5. While implementing the BERT model we found that the model was very heavy and took lot of time to
complete training. We could run it only on Linux machine that too in local Jupyter instance.

6. When we ran BERT model with ktrain in google collab, the RAM crashes even for 200 records. Hence we
could only manage to run BERT model with ktrain only for 100 records.

### Closing Reflections
Below are few improvement points that could be taken as further improvement points:
1. After language translation, there should be separate preprocessing done on the non-English texts.
2. There is one frequent word “SID_24” that was holding some meaningful information, could have been
restored differently. In current preprocessing we are removing all the digits.

3. Sampling of the data should have been experimented considering the data being highly imbalanced.

4. Different techniques of model deployment can be explored for better results and graphics.

5. Building Machine learning models on android.



## Authors

#### Shalini Tiwari
#### AnandaKrishna S
#### Gourav Saha
#### Vamsee Krishna
