import tensorflow as tf
from tensorflow_hub import KerasLayer
import csv
import numpy as np


# def load_our_model(model_path):
#     """
#     Loads model from input path
#     """
#     # "custom_objects" used to tell Keras that there
#     # is an additional (custom) layer (from training)
#     model = tf.keras.models.load_model(model_path,
#                        custom_objects={"KerasLayer": KerasLayer})
#     return model


# def get_unique_labels():
#     read_list = []
#     with open('/unique_labels.csv', 'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             read_list = row
#     return read_list


# # Turn prediction probabilities into the predicted label
# def get_pred_label(prediction_probabilites, unique_labels):
#     return unique_labels[np.argmax(prediction_probabilites)]
