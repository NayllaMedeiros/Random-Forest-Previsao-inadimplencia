# Machine Learning with Radom Forest for Default
## Introduction
In this project, my objective was to reduce a company's loan default rate. To do this, I first exported the data that was analyzed in SQL, cleaned it using Python and finally, I used a machine learning algorithm called Radom Forest to make predictions whether or not new customers would pay the loans.

## Random-Forest-Previsao-inadimplencia
Random forest is a supervised learning algorithm. It has two variations – one is used for classification problems and other is used for regression problems. It is one of the most flexible and easy to use algorithm. It creates decision trees on the given data samples, gets prediction from each tree and selects the best solution by means of voting. It is also a pretty good indicator of feature importance.

Random forest algorithm combines multiple decision-trees, resulting in a forest of trees, hence the name Random Forest. In the random forest classifier, the higher the number of trees in the forest results in higher accuracy.

## Requirements
* **Python 3.6+**
* **NumPy (`pip install numpy`)**
* **Pandas (`pip install pandas`)**
* **Scikit-learn (`pip install scikit-learn`)**
* **(`import statistics  as sts`)**
* **(`import seaborn as srn`)**
* **(`import numpy as np`)**
* **(`import psycopg2`)**
* **(`import pandas as pd`)**
* **('from sklearn.ensemble import RandomForestClassifier')**

## Export data from SQL to Python
1. Connect to database
conn = psycopg2.connect(
    dbname="database_name",
    user="user_name",
    password="user_password",
    host="localhost" # or your database address
)

2. Load data from the SQL table into a DataFrame
query = "SELECT * FROM table_name"
df = pd.read_sql_query(query, conn)

3. Close the database connection
conn.close()

4. Print the DataFrame
print(df)

## Data analysis and cleaning
1. Data Exploration Techniques in Python
2. Data Manipulation Using pandas
3. Statistcs for to clean

## Introduction to Random Forest
1. Introduction to Random Forest algorithm
2. Random Forest algorithm intuition
3. Advantages and disadvantages of Random Forest algorithm
4. Feature selection with Random Forests
5. The problem statement
6. Dataset description
7. Import libraries
8. Import dataset
9. Exploratory data analysis
10. Declare feature vector and target variable
11. Split data into separate training and test set
12. Feature engineering
13. Random Forest Classifier model with default parameters
14. Random Forest Classifier model with parameter n_estimators=100
15. Find important features with Random Forest model
16. Visualize the feature scores of the features
17. Build the Random Forest model on selected features
18. Confusion matrix
19. Results and conclusion

## Conclusion
In this project, I built a Random Forest Classifier to predict a company's loan default rate. After carrying out the training and testing, the result was a 79% success rate for this algorithm, which means that with the prediction of whether or not a new customer will be in default, they will be 79% sure of being correct.
