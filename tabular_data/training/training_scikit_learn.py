from sklearn.metrics import accuracy_score
import logging

def training_sklearn(model_dict, x_train, x_test, y_train, y_test):
    model_type = model_dict['type']
    model_name = model_dict['name']
    model = model_dict['model']

    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    y_pred_train = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    logging.warning(f"Final Train Accuracy: {train_accuracy:.4f} | Final Test Accuracy: {test_accuracy:.4f}")
    return {'type': model_type, 'name': model_name, 'model': model, 'final_train_acc': train_accuracy, 'final_test_acc': test_accuracy}


def training_sklearn_kernel(model_dict, x_train, x_test, y_train, y_test):
    model_type = model_dict['type']
    model_name = model_dict['name']
    model = model_dict['model']

    kernel_matrix_training, kernel_matrix_test = model.get_kernels(x_train, x_test)
    model.fit(kernel_matrix_training, y_train)
    y_pred_test = model.predict(kernel_matrix_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    y_pred_train = model.predict(kernel_matrix_training)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    logging.warning(f"Final Train Accuracy: {train_accuracy:.4f} | Final Test Accuracy: {test_accuracy:.4f}")
    return {'type': model_type, 'name': model_name, 'model': model, 'final_train_acc': train_accuracy, 'final_test_acc': test_accuracy}

