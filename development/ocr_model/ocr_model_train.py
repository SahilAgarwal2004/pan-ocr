import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
import wandb
from wandb.integration.keras import WandbMetricsLogger
import cv2
import os
from sklearn.model_selection import train_test_split
import string
from datetime import datetime
import sys

# Error handling for custom preprocessing import
try:
    from ocr_preprocess import preprocess_pan_image
except ImportError:
    print("Error: ocr_preprocess.py not found or preprocess_pan_image function not available")
    print("Please ensure ocr_preprocess.py exists with the preprocess_pan_image function")
    sys.exit(1)

def validate_requirements():
    required_files = ['master.csv', 'ocr_preprocess.py']
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    try:
        df = pd.read_csv('master.csv')
        required_columns = ['image_path', 'label']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ CSV missing required columns: {required_columns}")
            return False
        if len(df) == 0:
            print("❌ CSV file is empty")
            return False
        print(f"✅ Found {len(df)} records in master.csv")
    except Exception as e:
        print(f"❌ Error reading master.csv: {e}")
        return False
    return True

def initialize_wandb():
    try:
        wandb.init(
            project="pan-ocr-model",
            name=f"vgg-bilstm-final-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 100,
                "max_sequence_length": 32,
                "img_height": 64,
                "img_width": 256
            }
        )
        print("✅ Weights & Biases initialized successfully")
        return wandb.config
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize wandb: {e}")
        print("Continuing without wandb logging...")
        class DefaultConfig:
            learning_rate = 0.001
            batch_size = 16
            epochs = 100
            max_sequence_length = 32
            img_height = 64
            img_width = 256
        return DefaultConfig()

# Character mapping: 0-based for characters, blank is VOCAB_SIZE - 1 (CTC standard)
CHARACTERS = string.ascii_uppercase + string.digits + "/"
CHAR_TO_NUM = {char: idx for idx, char in enumerate(CHARACTERS)}
NUM_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}
VOCAB_SIZE = len(CHARACTERS) + 1  # 37 + 1 = 38; blank is 37

def setup_gpu():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(physical_devices)} GPU(s)")
            return True
        else:
            print("⚠️ No GPU detected, using CPU")
            return False
    except Exception as e:
        print(f"⚠️ GPU setup error: {e}")
        return False

def build_vgg_feature_extractor(img_height, img_width):
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    return Model(inputs, x)

def build_ocr_model(img_height, img_width):
    inputs = layers.Input(shape=(img_height, img_width, 3))
    vgg_features = build_vgg_feature_extractor(img_height, img_width)(inputs)
    new_shape = ((img_width // 8), (img_height // 8) * 512)
    x = layers.Reshape(target_shape=new_shape)(vgg_features)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    model = Model(inputs, x)
    print(f"✅ OCR model built successfully. Output sequence length: {model.output.shape[1]}")
    return model

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, name="ctc_loss"):
        super().__init__(name=name)
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        batch_len = tf.shape(y_true)[0]
        input_length = tf.shape(y_pred)[1]
        label_length = tf.reduce_sum(tf.cast(y_true != (VOCAB_SIZE - 1), tf.int32), axis=1)
        input_length = input_length * tf.ones(shape=(batch_len,), dtype=tf.int32)
        loss = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred,
            tf.reshape(input_length, (-1, 1)),
            tf.reshape(label_length, (-1, 1))
        )
        return loss

class CERMetric(tf.keras.metrics.Metric):
    def __init__(self, name='CER', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="cer_accumulator", initializer="zeros")
        self.counter = self.add_weight(name="counter", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_int = tf.cast(y_true, dtype=tf.int32)
        input_shape = tf.shape(y_pred)
        input_length = tf.ones(shape=(input_shape[0],), dtype=tf.int32) * input_shape[1]
        decoded_list, _ = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)
        decoded_dense = decoded_list[0]
        # Convert decoded output to SparseTensor, excluding blanks (-1)
        decoded_lengths = tf.reduce_sum(tf.cast(decoded_dense != -1, tf.int32), axis=1)
        hypothesis_sparse = tf.cast(
            tf.keras.backend.ctc_label_dense_to_sparse(decoded_dense, decoded_lengths), tf.int64
        )
        label_length = tf.reduce_sum(tf.cast(y_true_int != (VOCAB_SIZE - 1), tf.int32), axis=1)
        truth_sparse = tf.cast(
            tf.keras.backend.ctc_label_dense_to_sparse(y_true_int, label_length), tf.int64
        )
        distance = tf.edit_distance(hypothesis_sparse, truth_sparse, normalize=True)
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)
    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)

def preprocess_image(image_path, img_height=64, img_width=256):
    try:
        image = cv2.imread(image_path)
        if image is None: return None
        processed_image = preprocess_pan_image(image)
        if processed_image is None: return None
        if len(processed_image.shape) == 2:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        processed_image = cv2.resize(processed_image, (img_width, img_height))
        return processed_image.astype(np.float32) / 255.0
    except Exception as e:
        print(f"⚠️ Error preprocessing image {image_path}: {e}")
        return None

def encode_labels(labels, max_sequence_length):
    encoded_labels = []
    for label in labels:
        encoded = [CHAR_TO_NUM[char] for char in str(label).upper() if char in CHAR_TO_NUM]
        pad_width = max_sequence_length - len(encoded)
        # Pad with blank index (VOCAB_SIZE - 1)
        encoded.extend([VOCAB_SIZE - 1] * pad_width)
        encoded_labels.append(encoded)
    return np.array(encoded_labels, dtype=np.int32)

def load_data(csv_path, img_height, img_width, max_label_len):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(str)
    original_count = len(df)
    df = df[df['label'].str.len() <= max_label_len].copy()
    print(f"ℹ️ Filtered {original_count - len(df)} samples with labels longer than {max_label_len} characters.")
    images, labels = [], []
    for _, row in df.iterrows():
        image = preprocess_image(row['image_path'], img_height, img_width)
        if image is not None:
            images.append(image)
            labels.append(row['label'])
    images = np.array(images)
    encoded_labels = encode_labels(labels, max_sequence_length=max_label_len)
    print(f"✅ Successfully loaded {len(images)} images.")
    return images, encoded_labels

def create_callbacks(use_wandb=True):
    os.makedirs('./checkpoints', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('./checkpoints/best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
    ]
    if use_wandb:
        callbacks.append(WandbMetricsLogger())
    print(f"✅ Created {len(callbacks)} training callbacks.")
    return callbacks

def main():
    try:
        print("🚀 Starting OCR Model Training Pipeline...")
        if not validate_requirements(): return
        config = initialize_wandb()
        use_wandb = wandb.run is not None
        setup_gpu()
        model = build_ocr_model(img_height=config.img_height, img_width=config.img_width)
        max_len = model.output.shape[1]
        images, labels = load_data('master.csv', config.img_height, config.img_width, max_label_len=max_len)
        X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        print(f"📊 Dataset split: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        np.save('./checkpoints/X_test.npy', X_test)
        np.save('./checkpoints/y_test.npy', y_test)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss=CTCLoss(),
            metrics=[CERMetric()]
        )
        model.fit(
            X_train, y_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=(X_val, y_val),
            callbacks=create_callbacks(use_wandb),
            verbose=1
        )
        print("💾 Saving final model...")
        model.save('./checkpoints/final_model.h5')
        print("📊 Final evaluation on test set...")
        test_loss, test_cer = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Test Results - Loss: {test_loss:.4f}, CER: {test_cer:.4f}")
        if use_wandb:
            wandb.log({"test_loss": test_loss, "test_cer": test_cer})
            wandb.finish()
        print("🎉 Training completed successfully!")
    except Exception as e:
        print(f"❌ Fatal error in main training pipeline: {e}")
        import traceback
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
        sys.exit(0)
