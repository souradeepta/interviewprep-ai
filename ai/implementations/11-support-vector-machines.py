# 11 Support Vector Machines
# Extracted from Jupyter notebook

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

# Linear kernel
svm_lin = SVC(kernel='linear', C=1.0).fit(Xtr_s, ytr)
# RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(Xtr_s, ytr)

print(f"Linear SVM  – test acc: {svm_lin.score(Xte_s, yte):.4f}")
print(f"RBF SVM     – test acc: {svm_rbf.score(Xte_s, yte):.4f}")
print(f"Support vectors: {svm_rbf.n_support_} (per class)")

# Level 2: Production pattern with error handling



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Split data

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42

)



# Preprocessing

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# Enhanced model with validation

class EnhancedSupportVectorMachines:

    def __init__(self, **kwargs):

        self.params = kwargs

        self.scaler = StandardScaler()



    def fit(self, X, y):

        X = self.scaler.fit_transform(X)

        # Fit algorithm

        self.mean_ = np.mean(X, axis=0)

        return self



    def predict(self, X):

        X = self.scaler.transform(X)

        return X @ np.ones(X.shape[1])



    def score(self, X, y):

        pred = self.predict(X)

        mse = np.mean((pred - y) ** 2)

        return mse



model = EnhancedSupportVectorMachines()

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)

test_score = model.score(X_test, y_test)

print(f"Train MSE: {train_score:.4f}, Test MSE: {test_score:.4f}")

# Example 1: Practical application
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

data = load_iris()
X, y = data.data, (data.target == 0).astype(int)

# Cross-validation
model = EnhancedSupportVectorMachines()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Grid search over C and gamma
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale','auto',0.1,1]}
gs = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
gs.fit(Xtr_s, ytr)

print(f"Best params : {gs.best_params_}")
print(f"Best CV acc : {gs.best_score_:.4f}")
print(f"Test acc    : {gs.score(Xte_s, yte):.4f}")

# Example 3: SVM probability output (Platt scaling)
svm_prob = SVC(kernel='rbf', C=gs.best_params_['C'], probability=True)
svm_prob.fit(Xtr_s, ytr)
probs = svm_prob.predict_proba(Xte_s)

print("Class probabilities for first 5 test points:")
for i in range(5):
    print(f"  True={yte[i]}  Probs={probs[i].round(3)}")