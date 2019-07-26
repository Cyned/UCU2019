from sklearn.dummy import DummyClassifier

from metrics.cross_val import custom_metric


my_metric = custom_metric()


__all__ = [
    'my_metric',
]


if __name__ == '__main__':
    print(my_metric(classifier=DummyClassifier(random_state=134134)))
