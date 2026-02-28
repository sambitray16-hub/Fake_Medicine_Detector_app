import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(data_dir, model_save_path='models/medicine_classifier.h5'):
    """
    Trains a custom medicine classifier using Transfer Learning on MobileNetV2.
    
    Expects data_dir to have 'train' and 'val' subfolders, each with 'genuine' and 'fake' folders.
    """
    
    # 1. Setup Data Generators (with Augmentation for better fakes detection)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found. Please create it and add 'train' and 'val' folders.")
        return

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # 2. Build Model using Transfer Learning
    print("Building model with MobileNetV2 base...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    base_model.trainable = False

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x) # [Fake, Genuine]

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train
    print("Starting training...")
    # Note: Adjust epochs based on your dataset size
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    # 5. Save the trained model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model.save(model_save_path)
    print(f"Training complete! Custom model saved to {model_save_path}")

if __name__ == "__main__":
    # Pointing to the newly created dataset folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    
    print(f"Training using dataset at: {DATA_DIR}")
    train_model(DATA_DIR)
