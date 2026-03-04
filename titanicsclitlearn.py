import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load the dataf
df =pd.read_csv('Titanic-Dataset.csv')

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
plt.show()


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
plt.show()

# Show the new weights for the SCALED model
weights_scaled = pd.DataFrame({'Feature': X.columns, 'Weight': model_scaled.coef_[0]})
print(weights_scaled.sort_values(by='Weight', ascending=False))``