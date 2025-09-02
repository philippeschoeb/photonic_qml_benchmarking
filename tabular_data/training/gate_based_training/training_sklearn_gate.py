from sklearn.metrics import accuracy_score
import logging

def training_sklearn_gate(model_dict, x_train, x_test, y_train, y_test):
    model_type = model_dict['type']
    model_name = model_dict['name']
    model = model_dict['model']

    # Convert tensors to numpy arrays
    x_train = x_train.detach().cpu().numpy()
    x_test = x_test.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    y_pred_train = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    logging.warning(f"Final Train Accuracy: {train_accuracy:.4f} | Final Test Accuracy: {test_accuracy:.4f}")
    return {'type': model_type, 'name': model_name, 'model': model, 'final_train_acc': train_accuracy, 'final_test_acc': test_accuracy}
