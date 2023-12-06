# Imports
import numpy as np
from model import EfficientDecoder

# Dummy Caption - This should be NE (Numerical Encoded) Vectors of Words from Target Vocab
x = np.random.randn(1, 50) # (bs, seq_len)

# Load Image Encodings from NPZ files
file_path = "./MaxPooledFeatures/36979.npz"
image_encodings = np.load(file_path)
key = 'arr_0'

# Assert Shape
context = image_encodings[key]
assert context.shape == (1792, )

# Reshape for Model
context = np.tile(np.expand_dims(context, axis = 0), (1, 50, 1)) # (bs, seq_len, img_enc)

# Model Parameters
num_layers = 3
d_model = 512
num_heads = 4
dff = 2048
target_vocab_size = 1000

# Define Model
model = EfficientDecoder(num_layers = num_layers, d_model = d_model, num_heads = num_heads, target_vocab_size = target_vocab_size, dff = dff)
op = model((context, x))
print(op.shape)