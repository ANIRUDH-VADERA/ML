import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,classification_report
import seaborn as sns

# Importing the dataset
df=pd.read_csv("C:/Users/Anirudh/OneDrive/Desktop/emails.csv")
print("The dataset is as following : [5172 rows x 3002 columns]")
print(df)
print("\n")

# 0 - Non Spam
# 1 - Spam
df['Prediction'].value_counts(normalize=True)


df_train, df_test = train_test_split(df,test_size=0.009,train_size=0.991,random_state=0)

# Isolating spam and ham messages first
spam_messages = df_train[df_train['Prediction'] == 1]
ham_messages = df_train[df_train['Prediction'] == 0]

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(df_train)
p_ham = len(ham_messages) / len(df_train)

df_train_n_of_count_spam = spam_messages.iloc[:,1:-1]
df_train_n_of_count_ham = ham_messages.iloc[:,1:-1]
spam_messages['No_of_words']= df_train_n_of_count_spam.sum(axis=1)
ham_messages['No_of_words']= df_train_n_of_count_ham.sum(axis=1)

# N_Spam
n_spam = spam_messages['No_of_words'].sum()

# N_Ham
n_ham = ham_messages['No_of_words'].sum()

# N_Vocabulary
n_vocabulary = len(df_train.columns) - 2

# Laplace smoothing
alpha = 1

# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in df_train.columns[1:-1]}
parameters_ham = {unique_word:0 for unique_word in df_train.columns[1:-1]}

# Calculate parameters
for word in df_train.columns[1:-1]:
   n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
   p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
   parameters_spam[word] = p_word_given_spam

   n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
   p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
   parameters_ham[word] = p_word_given_ham

def classify(message):

   p_spam_given_message = p_spam
   p_ham_given_message = p_ham

   for word in message:
      if word in parameters_spam:
         p_spam_given_message *= parameters_spam[word]

      if word in parameters_ham: 
         p_ham_given_message *= parameters_ham[word]

   if p_ham_given_message > p_spam_given_message:
      return 0
   elif p_ham_given_message < p_spam_given_message:
      return 1
   else:
      return 0

message_list_to_predict = []
Y_pred = []
itr = 0
while(itr<len(df_test)):
    message_list_to_predict = []
    columns = df_test.columns[1:-1]
    for column in columns:
        temp = df_test.iloc[itr,:][column]
        for i in range(temp):
            message_list_to_predict.append(column)
    Y_pred.append(classify(message_list_to_predict))
    itr = itr + 1
    
Y_test = df_test.iloc[:,-1]

# Checking the accuracy of our model
print('Accuracy: ',accuracy_score(Y_test,Y_pred))
print('Precision: %.3f' % precision_score(Y_test, Y_pred))
print('Recall: %.3f' % recall_score(Y_test, Y_pred))


# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = "(Predicted and Actual Y_values)"
plt.title(all_sample_title, size = 15)

# likelihood Probabilities
print(parameters_spam)
print(parameters_ham)