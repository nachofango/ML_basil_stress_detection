import tensorflow as tf
from keras.src.trainers.data_adapters.data_adapter_utils import class_weight_to_sample_weights
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
# --- Imports for metrics, data handling, and visualization ---
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


# --- Directories & Configuration ---
# Base directory for folders: 'train' y 'val'
base_dir = '/DATA-PROCESSING/data'
target_size = (224, 224) # VGG16 requires input dimensions of (224, 224)
batch_size = 32


# --- NEW: Calculate Class Weights to balance 60/40 ratio between healthy and stressed classes ---
# Healthy: 564, Stressed: 364 (Total: 928)
total_samples = 928
num_classes = 2

# Standard formula: total / (n_classes * class_samples)
weight_for_0  = total_samples / (num_classes * 564) # ~0.82 (healthy)
weight_for_1  = total_samples / (num_classes * 364) # ~1.27 (stressed)

class_weights = {0: 0.82, 1: 1.27}


# ----------- Pre processing of data -----------

# 1. Training Generator (Normalization ONLY)
# Each pixel intensity was divided by 255, which is the maximum value of the input signal.
# This ensures that the input signals are on a consistent scale between 0 and 1,
# which helps the model converge efficiently during training.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Standardize pixel values to [0, 1] range
    rotation_range=20,  # Randomly rotate images up to 20 degrees
    width_shift_range=0.1,  # Horizontally shift images (10% of total width)
    height_shift_range=0.1,  # Vertically shift images (10% of total height)
    shear_range=0.1,  # Apply slant transformations
    zoom_range=0.2,  # Randomly zoom in/out by 20%
    horizontal_flip=True,  # Flip images to simulate different leaf orientations
    fill_mode="nearest"  # Strategy to fill empty pixels created by transformations
)

# 1. Validating Generator (Normalization ONLY)
val_datagen = ImageDataGenerator(rescale=1./255)

# --- Data flow ---
# Loads the training images (resizes them to 224x224)
# Creates iterable object that will feed the model
train_generator = train_datagen.flow_from_directory(
    base_dir + '/train',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
    #shuffle=True (by default) It randomizes the order of images in every epoch. This is vital to keep the model
    # from learning the sample order and to minimize overfitting.
)

# Loads the validation images (resizes them to 224x224)
validation_generator = val_datagen.flow_from_directory(
    base_dir + '/val',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # It ensures the model's predictions are matched with the labels loaded from the directory structure.
                  # If it was set to True the confusion matrix would not make sense
)

print("Images loaded in 'train_generator' and 'validation_generator'.")


# ------------Setting up the  feature extraction part of the VGG16 model to learn from scratch ---------------------
base_model = VGG16( #was used to define the standard VGG16 architecture.
    weights=None,                        # To train the model from scratch, the architecture was initialized with
                                         # random weights to learn features from the dataset.
    include_top=False,                   # Excludes the final 1000-class ImageNet classifier to add a custom binary head
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = True # Iterates through the base model layers and marks them as trainable parameters

#----------------- Customization of the models "head"--------------------------------
model = models.Sequential([
    base_model,
    layers.Flatten(),       #The Flatten layer converts the output of the base model into a single 1D vector to
                            # be connected to the fully connected (Dense) classification layers.

    layers.Dense(64, activation='relu'),        # is the initial layer of the custom classification head, which
                                                      # takes the output of the Flatten layer and converts it into 64
    # distinct numerical values. It uses 64 neurons and the ReLU activation function to learn non-linear combinations
    # of the features extracted by the VGG16 base model.

    layers.Dense(2, activation='softmax')     # this is the output layer of the model. It has 2 neurons
    # corresponding to the 2 classes, “healthy” and “stressed”. It uses the Softmax activation function,
    # which converts the outputs from the previous layers into a probability distribution where the sum of probabilities
    # for all classes equals 1. For this model it means the result of the Softmax activation function would look like
    # this: [0.10, 0.90], meaning the model is 90% confident that the image belongs to the “stressed” class.
])


# ---------------- Model compilation ------------------------------------------------
print("Compiling the model...")
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9), # optimization algorithm
    loss='categorical_crossentropy',   # cost function used in the original VGG setup
    metrics=['accuracy']     # Calculates the percentage of correct guesses at the end of each epoch.
                             # This metric is used by the ModelCheckpoint to identify and save the "best" version.
)
model.summary()

# --- Define Callback (before model.fit) ---

# 1. Define the filename for the best model.
checkpoint_filepath = 'best_VGG16_scratch_val_acc.h5'

# 2. Define the ModelCheckpoint callback itself
# ModelCheckpoint used for saving the best version of the model based on a monitored metric.
# It checks the val_accuracy at the end of each epoch, and it saves the weights whenever a new best value is achieved.
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1                       # Show a message when a new best model is saved.
)

# Training of the model
print("Initiating training...")
history = model.fit(   # will report loss by default
    train_generator,
    epochs=250,
    validation_data=validation_generator,
    callbacks=[model_checkpoint_callback],
    class_weight=class_weights # This forces the 50/50 balance mathematically
)
print("Training finalized.")


# ---------------------Metric calculation (Precision, Recall, F1-Score and Confusion Matrix) ----------------------
print("\n=======================================================")
print("--- 1. CLASSIFICATION REPORT: LAST EPOCH MODEL ---")
print("=======================================================")

# Calculate the necessary steps for validation (to ensure model.predict covers the entire set)
val_steps = validation_generator.samples // validation_generator.batch_size + 1   # Calculates the number of batches needed to cover all 395 images.
                                                                                  # The '+ 1' ensures the final partial batch of 11 images is included.
y_pred_probs_last = model.predict(validation_generator, steps=val_steps)   # This line executes inference on the trained model.
                                                                           # y_pred_probs_last is a 2D array, a probability matrix. For each image, you get two values that sum to 1 (e.g., [0.10, 0.90]).

# Convert output probabilities to class labels (0 or 1)
y_pred_labels_last = np.argmax(y_pred_probs_last, axis=1)
# np.argmax selects the index (0 or 1) with the highest probability for each image.
# Converts the probability pairs (e.g., [0.2, 0.8]) into a single class index (0 or 1, healthy or stressed)

# Get the true labels and trim them to the size of the predictions
y_true_labels = validation_generator.classes     # This array obtains the right answers (labels 0 and 1) for every validation image
y_true_labels = y_true_labels[:len(y_pred_labels_last)]




#------------------------ Generate the report---------------------------------------
target_names = list(validation_generator.class_indices.keys()) # Converts the dictionary keys (folder names) into a list to provide human-readable labels for the final report.
print(classification_report(y_true_labels, y_pred_labels_last, target_names=target_names)) # Generates a detailed statistical report (precision, recall, f1-score) for each class.

#--------------------- Use Pandas to save history to CSV----------------------------
hist_df = pd.DataFrame(history.history)  # dictionary generated by keras after model.fit() is executed. it is an object with the full registry of the model during training
hist_csv_file = 'training_history_VGG16_scratch.csv'
hist_df.to_csv(hist_csv_file, index=False)
print(f"Training history saved to '{hist_csv_file}'")

#---------------------- Show confusion matrix---------------------------------------
print("\n--- CONFUSION MATRIX (Last Epoch Model) ---")
print(confusion_matrix(y_true_labels, y_pred_labels_last))                           # sklearn function

#----------- Load BEST ACCURACY Model and Calculate Metrics -------------------------

# Define the path to the best model file
best_model_path = 'best_VGG16_scratch_val_acc.h5'

try:
    # --------------- LOAD THE BEST MODEL ------------------------
    best_model = tf.keras.models.load_model(best_model_path)
    print(f"\nSuccessfully loaded model from '{best_model_path}'")

    print("\n=======================================================")
    print("--- 2. CLASSIFICATION REPORT: BEST ACCURACY MODEL ---")
    print("=======================================================")

    # Use the 'best_model' for prediction
    y_pred_probs_best = best_model.predict(validation_generator, steps=val_steps)

    # Convert output probabilities to class labels (0 or 1)
    y_pred_labels_best = np.argmax(y_pred_probs_best, axis=1)

    # Generate the report (using the same y_true_labels as before)
    print(classification_report(y_true_labels, y_pred_labels_best, target_names=target_names))

    # Show confusion matrix
    print("\n--- CONFUSION MATRIX (Best Validation Accuracy Model) ---")
    print(confusion_matrix(y_true_labels, y_pred_labels_best))



except Exception as e:
    print(f"\nERROR: Could not load the best model from '{best_model_path}'.")
    print(f"Details: {e}")
    print("Skipping Best Accuracy Model report.")


# ----------------- Plot History (Accuracy) -----------------------------------
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy of Training and Validation per Epoch (VGG16 from Scratch)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('accuracy_plot_VGG16_scratch.png')
plt.close()
print("Accuracy plot saved as 'accuracy_plot_VGG16_scratch.png'")


#--------------- Plot History (Loss) -------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Loss of training')
plt.plot(history.history['val_loss'], label='Loss of validation')
plt.title('Loss of Training and Validation per EPOCH (VGG16 from Scratch)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss_plot_VGG16_scratch.png') # Cambiamos el nombre
plt.close()
print("Loss graph saved as 'loss_plot_VGG16_scratch.png'")

