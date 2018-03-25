
# Dataset : Churn_Modelling.csv
This dataset that I'm using is of a fictional-bank, but it is pretty realistic. It has some observations like customerID, credit-score, gender, Salary etc. of 10k customers. The bank has been seeing unusual churn rates (churn is when people leave the company at unusually high rates) and they want to understand what the problem is. So they want to assess and address that problem.

So let's assume that the company hired me to look into this dataset for them and give them some insights. I've got a sample of 10,000 customers. This fictional bank operates in 3 countries : France, Spain and Germany. They have recorded some observations related to these customers like customerID, age, tenure, balance etc as mentioned earlier. So 6 months ago, they measured all these things and decided to watch them. And they're going to check who of these customers left the bank. In the dataset, 1 value of 'Exited' indicates the customer left the bank and 0 indicates that he's still with the bank. It's pretty obvious that this is a classification problem. 

My goal is to create a job demographic segmentation model to tell the bank which of the customers are at highest risk of leaving with the use of **ANNs**. 

# Results 

Epoch 100/100
8000/8000 [==============================] - 1s 118us/ step - loss: 0.3798 - acc: 0.8400

The accuracy (when I tested the model on test set) converged to 84% after the last(100th) epoch.

Confusion Matrix :

1521  74

242   163

Accuracy = 84.2%, which is pretty good considering the fact that I didn't use any kind of parameter tuning.

