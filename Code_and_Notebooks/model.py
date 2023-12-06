### Author: Jainam Shah

# Imports
from utils import Decoder
import tensorflow as tf

# Name of the Model - EfficientDecoder
class EfficientDecoder(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate = 0.1):
    super().__init__()

    # Dense layer for mapping image encodings to d_model
    self.dense = tf.keras.layers.Dense(d_model, activation = 'relu')

    # Only utilize Decoder part of the Transformer Model, will be using EfficientNetB4 as Image Encoder explicitly
    # For training, we have already extracted image encodings and store as np arrays to save computations.
    # During inference, we will build pipeline: Inputs -> EfficientNetB4 -> EfficientDecoder -> Output
    self.decoder = Decoder(num_layers = num_layers, d_model = d_model,
                           num_heads = num_heads, dff = dff,
                           vocab_size = target_vocab_size,
                           dropout_rate = dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    # context is image encodings (bs, seeq_len, img_enc), x is decoder input sequence (bs, seq_len)
    context, x  = inputs
    context = self.dense(context) # (bs, seq_len, d_model)
    x = self.decoder(x, context)  # (bs, seq_len, d_model)
    logits = self.final_layer(x)  # (bs, seq_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits