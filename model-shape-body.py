import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# تعیین تنظیمات
input_shape = (150, 150, 3)  # اندازه ورودی تصاویر
num_classes = 10
batch_size = 26
epochs = 50

# آماده‌سازی داده‌ها
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'evaluation_dataset',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    'evaluation_dataset',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# ساخت مدل
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# کامپایل مدل
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# محاسبه تعداد گام‌ها در هر دوره آموزشی
steps_per_epoch = train_generator.samples // batch_size

# آموزش مدل
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# ذخیره مدل
model.save('body_shape_detection_model2.h5')
