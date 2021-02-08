import time
import tensorflow as tf
import numpy as np


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


# Use the function below to test model accuracy. It is good practice to take a new image in various scenerios for
# testing.
def predict_from_image(model, img_path, threshold=0.5):
    """
    Takes an image (.jpg, .jpeg, etc.) and checks if path is blocked by comparing to a confidence threshold.
    Default is set to 0.5.

    :param model: Keras model to be tested
    :param img_path: Path of image to be predicted as "blocked" or "free" using trained model.
    :param threshold:
    :return: predicted value from [0, 1]
    """
    img = tf.keras.preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(224, 224))
    img_arr = tf.keras.preprocessing.image.img_to_array(img=img)
    img_arr = np.array([img_arr]) * 1.0 / 255
    predict = model.predict(img_arr)
    if predict <= threshold:
        print("Blocked")
    else:
        print("Free")
    return predict


# It is good practice to check that predictions can be made efficiently for real-time processing. Predictions should
# be tested on the robot itself.
def test_predict_speed(model, img_path, loop_count=50):
    """
    Tests the prediction speed of model. Model may have to be loaded first using tf.keras.models.load_model() method.

    :param model: Keras model to be tested
    :param img_path: Points to the path of image to be tested for prediction speed.
    :param loop_count: Number of loops to iterate over. Larger values provide better convergence to real-time value.
    :return:
    """
    for i in range(loop_count):
        start_time = time.time()
        prediction = predict_from_image(model, img_path)
        print(time.time() - start_time)
