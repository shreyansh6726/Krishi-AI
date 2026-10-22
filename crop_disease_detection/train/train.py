import os
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# STEP 1: Locate Dataset in the Local KaggleHub Cache
# ==========================================
user_cache = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets")
shrey_cache = r"C:\Users\shrey\.cache\kagglehub\datasets"
kagglehub_cache = user_cache if os.path.exists(user_cache) else shrey_cache
dataset_root = os.path.join(
    kagglehub_cache,
    "vipoooool",
    "new-plant-diseases-dataset",
    "versions",
)
versions = [
    os.path.join(dataset_root, version)
    for version in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, version))
]
if not versions:
    raise FileNotFoundError(
        f"No cached plant disease dataset found under '{dataset_root}'."
    )
path = max(versions, key=os.path.getmtime)
print("Using cached dataset:", path)

# Dynamically locate 'train' and 'valid' folders inside the downloaded path
train_dir, valid_dir = None, None
for root, dirs, files in os.walk(path):
    if "train" in dirs and "valid" in dirs:
        train_dir = os.path.join(root, "train")
        valid_dir = os.path.join(root, "valid")
        break

# Fallback path lookup if structure is nested deeply
if not train_dir or not valid_dir:
    for root, dirs, files in os.walk(path):
        if os.path.basename(root) == "train" and not train_dir:
            train_dir = root
        if os.path.basename(root) == "valid" and not valid_dir:
            valid_dir = root

print(f"Train Directory Found: {train_dir}")
print(f"Validation Directory Found: {valid_dir}")

if not train_dir or not valid_dir:
    raise FileNotFoundError(
        f"Cached dataset '{path}' must contain both 'train' and 'valid' directories."
    )


# ==========================================
# STEP 2: Load and Preprocess Images
# ==========================================
IMG_SIZE = 128  # Resizing all leaf images to 128x128 pixels
BATCH_SIZE = 32 # Number of images processed simultaneously in memory

print("\nLoading training dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int'
)

print("Loading validation dataset...")
valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='int'
)

# Retrieve category names (all 38 plant and disease labels)
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Total Disease Classes Found: {num_classes}")

# Normalization layer: Scales raw pixel intensity values from [0, 255] down to [0.0, 1.0]
normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (normalization_layer(x), y))

# Optimize data pipeline loading speed using caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
valid_dataset = valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# ==========================================
# STEP 3: Design the Custom CNN Architecture
# ==========================================
print("\nBuilding Custom CNN architecture from scratch...")
model = models.Sequential([
    # Input layer matching image dimensions and 3 color channels (RGB)
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Convolutional Block 1
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional Block 2
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional Block 3
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional Block 4
    layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten 2D feature maps into a 1D feature vector
    layers.Flatten(),
    
    # Dense Fully Connected Layers
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), # Dropout regularization to prevent overfitting
    
    # Output Layer: Softmax activation outputs probabilities across all 38 classes
    layers.Dense(num_classes, activation='softmax')
])

# Print summary of the model layers and parameters
model.summary()


# ==========================================
# STEP 4: Compile the Model
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
print("\nModel compiled successfully!")


# ==========================================
# STEP 5: Train and Save the Model
# ==========================================
EPOCHS = 10  # Number of full passes through the dataset (increase later if needed)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "..", "model")
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "crop_disease_cnn_model.keras")

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_best_only=False,
    verbose=1
)

print(f"\nStarting model training for {EPOCHS} epochs...")
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

model.save(model_path)
print(f"\nTraining complete! Custom CNN model successfully saved at: {model_path}")