import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


def train_model(dataset_file, model_file):
    # Cargar dataset.
    with open(dataset_file, 'rb') as f:
        data_dict = pickle.load(f)

    data = data_dict['data']
    labels = [int(label) for label in data_dict['labels']]  # Convertir etiquetas a enteros.

    # Dividir en conjuntos de entrenamiento y prueba.
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    # Convertir y_train y y_test a enteros (por seguridad).
    y_train = [int(label) for label in y_train]
    y_test = [int(label) for label in y_test]

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    # Train XGBoost model with grid search.
    grid_search = GridSearchCV(
        XGBClassifier(),
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1 # todos los nucleos.
    )
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    print("Training complete!")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Overall Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

    # Guardar el modelo entrenado.
    with open(model_file, 'wb') as f:
        pickle.dump({'model': best_model}, f)

    print(f"Model saved to {model_file}")


if __name__ == '__main__':
    dataset_path = './data.pickle'
    model_output = './model.p'
    train_model(dataset_path, model_output)
