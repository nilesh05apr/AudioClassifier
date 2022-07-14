from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings('ignore')


def ANN(input_shape):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def model_list()->dict:
    models = dict({'logreg': LogisticRegression(solver='lbfgs', max_iter=1000,C=0.2),
            'sgd': SGDClassifier(loss='modified_huber',random_state=101),
            'knn': KNeighborsClassifier(),
            'dtc': DecisionTreeClassifier(max_depth=10, random_state=42),
            'svm': SVC(kernel='linear',C=0.025,random_state=101),
            'rfc': RandomForestClassifier(max_depth=10, random_state=101),
            'mlp': MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)})
    return models