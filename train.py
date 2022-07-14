import pandas as pd
from models import ANN,model_list
#from data import init_data
from datetime import datetime 
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint


models = model_list()

def train_ann(X_train,X_test,y_train,y_test):
    model = ANN(X_train.shape[1])
    num_epochs = 15
    num_batch_size = 32
    checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                                verbose=1, save_best_only=True)
    start = datetime.now()
    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    test_accuracy=model.evaluate(X_test,y_test,verbose=0)
    print(test_accuracy[1])
    return test_accuracy


def train(name,X_train,X_test,y_train,y_test):
    results = []
    names = []
    dfs=[]
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['Healthy', 'Non-Healthy']
    model = models[name]
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
    cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(name)
    print(classification_report(y_test, y_pred, target_names=target_names))
    results.append(cv_results)
    names.append(name)
    this_df = pd.DataFrame(cv_results)
    this_df['model'] = name
    dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)
    return final



def train_all(X_train,X_test,y_train,y_test):
    results = dict()
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['Healthy', 'Non-Healthy']
    dfs = []
    for name, model in models.items():
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        results[name] = classification_report(y_test, y_pred, target_names=target_names)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)
    return final,results
