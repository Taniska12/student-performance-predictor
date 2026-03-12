import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("student_data.csv")

# Convert categorical columns to numbers
data['Parental_Involvement'] = data['Parental_Involvement'].astype('category').cat.codes
data['Distance_from_Home'] = data['Distance_from_Home'].astype('category').cat.codes
data['Gender'] = data['Gender'].astype('category').cat.codes

# Features (inputs)
X = data[['Hours_Studied','Attendance','Parental_Involvement','Distance_from_Home','Gender']]

# Target (output)
y = data['Exam_Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create model
model = RandomForestRegressor()

# Train model
model.fit(X_train,y_train)

# Save model
pickle.dump(model,open("model.pkl","wb"))

print("Model trained successfully and saved as model.pkl")