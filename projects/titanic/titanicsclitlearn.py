import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load the dataf
df =pd.read_csv('../../data/Titanic-Dataset.csv')

# 2. Drop columns that are "Noise" (Names and Ticket numbers don't help predict survival)
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 3. Handle Missing Values
# Fill missing Age with the median
df['Age'] = df['Age'].fillna(df['Age'].median())
# Fill missing Embarked with the most common value (the 'mode')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 4. Convert Text to Numbers (Encoding)
# 'Sex' becomes 0 or 1. 'Embarked' gets split into 3 columns (C, Q, S)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 5. Define Features (X) and Target (y)
X = df.drop('Survived', axis=1) # Everything except the answer
y = df['Survived']              # The answer we want to predict

# 6. The Tuesday Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data is ready! X_train shape:", X_train.shape)
print(df.head())



# 1. Initialize the model 
# (max_iter=1000 ensures the solver has enough time to find the best weights)
model = LogisticRegression(max_iter=1000)

# 2. Fit the model (The "Learning" phase)
model.fit(X_train, y_train)

# 3. Make predictions on the test set
y_pred = model.predict(X_test)

# 4. Check the Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# This shows [Probability of Death, Probability of Survival]
probabilities = model.predict_proba(X_test)
print(probabilities[:5])

# Create a DataFrame to see weights next to feature names
weights = pd.DataFrame({'Feature': X.columns, 'Weight': model.coef_[0]})
print(weights.sort_values(by='Weight', ascending=False))




#-----------------------------


# 1. Get the predictions for the test set

# 2. Create the matrix
cm = confusion_matrix(y_test, y_pred)

# 3. Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Died', 'Survived'])
disp.plot(cmap='Blues')
#plt.show()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_scaled = LogisticRegression()
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)
cm_scaled = confusion_matrix(y_test, y_pred_scaled)
disp_scaled = ConfusionMatrixDisplay(confusion_matrix=cm_scaled, display_labels=['Died', 'Survived'])

disp_scaled.plot(cmap='Greens')
#plt.show()

# Show the new weights for the SCALED model
weights_scaled = pd.DataFrame({'Feature': X.columns, 'Weight': model_scaled.coef_[0]})
print(weights_scaled.sort_values(by='Weight', ascending=False))


# use randomforestclassifier to do complex if/else splits and see if we can get better accuracy than logistic regression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Instantiate the "Forest" (n_estimators = 100 decision trees)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train the model (Notice we are using the original, unscaled X_train)
rf_model.fit(X_train, y_train)

# 3. Predict on the test set
rf_pred = rf_model.predict(X_test)

# 4. Check the new Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# 5. Show the new Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Died', 'Survived'])
disp_rf.plot(cmap='Purples')
plt.title("Random Forest Confusion Matrix")
plt.show()

# 1. Extract the importance scores from the model
importances = rf_model.feature_importances_

# 2. Match them to your column names
forest_weights = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# 3. Sort them from lowest to highest for the chart
print(forest_weights.sort_values(by='Importance', ascending=True))

