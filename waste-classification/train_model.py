import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

def create_waste_model():
    """Create CNN model for 3-class waste classification"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes: biodegradable, recyclable, trash
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data():
    """Prepare training and validation data"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # Use 20% for validation
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation data
    validation_generator = validation_datagen.flow_from_directory(
        'dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

def train_model():
    """Train the waste classification model"""
    print("🚀 Starting model training...")
    
    # Create model
    model = create_waste_model()
    model.summary()
    
    # Prepare data
    train_gen, val_gen = prepare_data()
    
    print(f"📊 Found {train_gen.samples} training images")
    print(f"📊 Found {val_gen.samples} validation images")
    print(f"🏷️ Classes: {list(train_gen.class_indices.keys())}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        epochs=25,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('waste_classifier.h5')
    print("✅ Model saved as waste_classifier.h5")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("📈 Training history saved as training_history.png")

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('dataset'):
        print("❌ Dataset folder not found!")
        print("Make sure you have the dataset folder with subdirectories:")
        print("- dataset/biodegradable/")
        print("- dataset/recyclable/")
        print("- dataset/trash/")
    else:
        print("✅ Dataset found!")
        model, history = train_model()