# Import necessary libraries
import tensorflow as tf  # TensorFlow for deep learning
import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for displaying images
import PIL.Image  # PIL (Pillow) for image loading and processing

# Load pre-trained VGG19 model (without the fully connected layers)
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Define layers that will be used to extract style and content features
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layer = 'block5_conv2'

def load_and_process_image(image_path):
    """
    Loads an image, resizes it, and preprocesses it for the VGG19 model.
    """
    img = PIL.Image.open(image_path)  # Open the image file
    img = img.resize((512, 512))  # Resize the image to 512x512 pixels
    img = np.array(img, dtype=np.float32)  # Convert image to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 512, 512, 3)
    img = tf.keras.applications.vgg19.preprocess_input(img)  # Normalize image for VGG19
    return img

def deprocess_image(img):
    """
    Converts a processed image back to normal display format.
    """
    img = img.reshape((512, 512, 3))  # Reshape image to original size
    img[:, :, 0] += 103.939  # Add VGG19 mean values back for R channel
    img[:, :, 1] += 116.779  # Add mean values for G channel
    img[:, :, 2] += 123.68  # Add mean values for B channel
    img = np.clip(img, 0, 255).astype('uint8')  # Clip values and convert to uint8
    return img

def get_model_outputs(image):
    """
    Extracts style and content features from an image using VGG19.
    """
    # Get specific layer outputs
    outputs = [vgg.get_layer(name).output for name in style_layers + [content_layer]]
    
    # Define a new functional model to extract features
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)  
    
    # Convert image to TensorFlow tensor (to avoid warnings)
    image = tf.convert_to_tensor(image, dtype=tf.float32)  
    
    return model(image)  # Pass the image through the model

# Load content and style images
content_image = load_and_process_image("content.jpg")  # Replace with your content image
style_image = load_and_process_image("style.jpg")  # Replace with your style image

# Extract content and style features
content_features = get_model_outputs(content_image)
style_features = get_model_outputs(style_image)

# Display original content and style images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Show the content image
ax1.imshow(PIL.Image.open("content.jpg"))  
ax1.set_title("Content Image")
ax1.axis("off")

# Show the style image
ax2.imshow(PIL.Image.open("style.jpg"))  
ax2.set_title("Style Image")
ax2.axis("off")

plt.show()
