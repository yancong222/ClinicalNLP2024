import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn import metrics
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

data = ('/your data folder/')
figures = ('/save the figures here/')

chinese = pd.read_csv(data + 'Chinese.csv', index_col=0)
chinese_matched = chinese[(chinese['gpt2_surp_mean'].notna()) &
                  (chinese['bert_surp_mean'].notna()) &
                  (chinese['llama_surp_mean'].notna())]
chinese_matched = chinese_matched.reset_index(drop=True)

"""# logistic regression
## predict group
"""

chinese_matched = chinese_matched[chinese_matched.group.isin(['PWA', 'HC'])]
chinese_matched.group.value_counts()

# Load the diabetes dataset
X, y = chinese_matched[['clean_utterance_len',
                'gpt2_surp_mean',
                'bert_surp_mean',
                'llama_surp_mean']], chinese_matched.group 
y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)

X_test

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print(np.unique(y_test))

le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)

# Plot ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
		label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
	accuracy * 100))
plt.legend(loc="lower right")
plt.savefig(figures + 'your figures.png', dpi=350)

plt.show()

y_test_encoded

"""## predict severity linear regression"""

chinese_pwa = chinese[chinese.group.isin(['PWA'])]
chinese_pwa = chinese_pwa[chinese_pwa['WAB-AQ'].notna()]
chinese_pwa = chinese_pwa.reset_index(drop = True)
len(chinese_pwa.Participant.unique())

# Load the diabetes dataset
X, y = chinese_pwa[['clean_utterance_len',
                'gpt2_surp_mean',
                'bert_surp_mean',
                'llama_surp_mean']], chinese_pwa['WAB-AQ']
y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)

X_test

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# retrieve the intercept
print(regressor.intercept_)

# retrieving the slope (coefficient of x)
print(regressor.coef_)

y_pred = regressor.predict(X_test)
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

pred_df

print('R squared:', metrics.r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

"""## predicting severity Elastic Net
"""

# Split the data into features and target
features = chinese_pwa[['clean_utterance_len',
                'gpt2_surp_mean',
                'bert_surp_mean',
                'llama_surp_mean']]

target = chinese_pwa['WAB-AQ']

 # Fit the ElasticNet model
model = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(features, target)
# Print the model coefficients
print(model.coef_)

#Implementation of ElasticNet
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)

# using repeated 10-fold cross-validation
# and report the average mean absolute error (MAE) on the dataset.
print("Root Mean Squared Error (ElasticNet): ",
      np.sqrt(-cross_val_score(elastic, X, y, cv=10,
                               scoring='neg_mean_squared_error')).mean())

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(elastic, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.abs(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# define model
ratios = np.arange(0, 1, 0.01)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]

model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
# fit model
model.fit(features, target)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)
print('l1_ratio_: %f' % model.l1_ratio_)

elastic = ElasticNet(alpha=0.1, l1_ratio=0)

print("Root Mean Squared Error (ElasticNet): ",
      np.sqrt(-cross_val_score(elastic, X, y, cv=10,
                               scoring='neg_mean_squared_error')).mean())

"""## predict types"""

chinese.WABtype.value_counts()
chinese_subtype = chinese[chinese.WABtype.isin(['AA (anomia)', 'Broca\'s'])] #, 'TMA (Tcmotor)' vs. anomia accuracy 51.28%
chinese_subtype.WABtype.value_counts()

# Select 194 row at random for each distinct value in column a.
# The random_state argument can be used to guarantee reproducibility:
chinese_subtype = chinese_subtype.groupby("WABtype").sample(n=86, random_state=1)
chinese_subtype.WABtype.value_counts()

X, y = chinese_subtype[['clean_utterance_len',
                'gpt2_surp_mean',
                'bert_surp_mean',
                'llama_surp_mean']], chinese_subtype.WABtype #['WAB-AQ']
y

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)

X_test

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)

# Plot ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
		label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
	accuracy * 100))
plt.legend(loc="lower right")
plt.savefig(figures + 'your figure.png', dpi=350)

plt.show()
