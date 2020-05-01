import glob
import os
import random
from operator import itemgetter
import string

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, KFold
from utils.plot import plot_cm, plot_graph

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model():
    """
    :param path: weight path
    :return: load model based on the path
    """
    path = latest_modified_weight()

    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def dump_model(model, name):
    model_name = WEIGHTS_DIR + name + generate_model_name(5) + '.pkl'
    with open(model_name, 'wb') as f:
        joblib.dump(value=model, filename=f, compress=3)
        print(f'Model saved at {model_name}')


def generate_model_name(size=5):
    """
    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def train_model(features, labels, args):
    if args.cv:

        cv_results = cross_validate(RandomForestClassifier(n_estimators=100, max_features='sqrt', criterion='entropy'),
                                    features, labels,
                                    cv=KFold(10, True, 42),
                                    return_estimator=True)

        estimator_testscore = zip(list(cv_results['estimator']), cv_results['test_score'])
        model = max(estimator_testscore, key=itemgetter(1))[0]
        preds = model.predict(features)

        score = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        plot_cm(cm, f'cm-accuracy:{score:.2f}RandomForest-cv')

        plot_graph(model.estimators_[0], features, labels, 'RandomForest-graph-cv')

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model.estimators_[0], 'RandomForest-cv-')

        return model.estimators_[0]
    else:

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

        model = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
        model.fit(features_train, labels_train)

        preds = model.predict(features_test)

        cm = confusion_matrix(labels_test, preds)
        score = accuracy_score(labels_test, preds)
        plot_cm(cm, f'cm-accuracy:{score:.2f}RandomForest')

        plot_graph(model.estimators_[0], features, labels, 'RandomForest-graph')

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model.estimators_[0], 'RandomForest')

        return model.estimators_[0]
