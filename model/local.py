from helpers import export_as_gif
import tensorflow as tf
import keras_cv
import time
from PIL import Image

# Uncomment this when utilizing GPU acceleration 
# Enables encoding vectors with 32 point precision and running model with 16 point precision
# tf.keras.mixed_precision.global_policy("mixed_float16")

start = time.time()

# set jit_compile true when running on anything other than mac
model = keras_cv.models.StableDiffusionV2(jit_compile=False)

# define prompts to guide diffusion process
prompt_1 = 'Ukiyo-e painting of a samurai in a poised stance'
prompt_2 = 'Ukiyo-e painting of a samurai in battle'

# convert prompts into a latent text encodings
prompt_encoding1 = tf.squeeze(model.encode_text(prompt_1))
prompt_encoding2 = tf.squeeze(model.encode_text(prompt_2))

# define a constant diffusion noise to maintain stability between the images generated
seed = 12345
noise = tf.random.normal((512 // 8, 512 // 8, 4), seed=seed)
interpolation_steps = 5

# create interpolation between the latent text encodings in the previous step
interpolated_encodings = tf.linspace(prompt_encoding1, prompt_encoding2, interpolation_steps)

encoding_finished = time.time()
print('\nEncoding took ', round(encoding_finished - start, 2), 's\n')

# run the model
images = model.generate_image(interpolated_encodings, batch_size=interpolation_steps, diffusion_noise=noise)

export_as_gif(
    "sumari.gif",
    [Image.fromarray(img) for img in images],
    frames_per_second=2,
    rubber_band=True
)

print('\nImage generation took ', round(time.time() - encoding_finished, 2), 's')

