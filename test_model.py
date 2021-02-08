# Testing model on new image. See utilities.py for more information.
import utilities
import os

save_path = os.path.join(os.getcwd(), 'model_save')  # specify model save path here

model = utilities.load_model(save_path)

img_path = '3ef7e5ba-6577-11eb-b08f-74d83e442a14.jpg'
prediction = utilities.predict_from_image(model, img_path)

utilities.test_predict_speed(model, img_path)
