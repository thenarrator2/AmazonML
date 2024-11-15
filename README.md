

# Entity Extraction and Classification

This project is an image and text-based entity extraction and classification tool that processes a dataset containing links to images, textual information, and associated entities. The system prepares and preprocesses the data, trains a model, and evaluates it to classify entities based on visual and textual input.

## Table of Contents
1. [Requirements](#requirements)
2. [Dataset Structure](#dataset-structure)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Model Architecture](#model-architecture)
5. [Usage](#usage)
6. [Evaluation](#evaluation)
7. [Results](#results)

### Requirements

- **Programming Language**: Python 3.8+
- **Libraries**:
  - TensorFlow
  - scikit-learn
  - pandas
  - numpy
  - requests
  - Pillow
  - seaborn
  - matplotlib

Install dependencies using:
```bash
pip install tensorflow scikit-learn pandas numpy requests pillow seaborn matplotlib
```

### Dataset Structure

The dataset should be a CSV file with the following required columns:
- `image_link`: URL link to an image.
- `group_id`: Identifier for grouping entities.
- `entity_name`: The name of the entity.
- `entity_value`: The value associated with the entity.
- `text`: Text description associated with the image.

Example:
```
image_link,group_id,entity_name,entity_value,text
https://example.com/image1.jpg,1,EntityA,ValueA,Sample text A
https://example.com/image2.jpg,2,EntityB,ValueB,Sample text B
```

### Preprocessing Steps

1. **Data Loading**: The `load_data()` function loads the dataset, checks for required columns, and returns a DataFrame.
2. **Image Processing**: Each image is resized to 224x224 pixels, normalized, and converted to an array for model input.
3. **Text Tokenization**: The text is tokenized and converted to padded sequences.
4. **Entity Encoding**: `encode_entities()` assigns numeric values to entity names.
5. **Data Preparation**: Data is split into training and test sets and saved as `.npy` files.

### Model Architecture

The model uses a dual-input structure:
- **Image Branch**: Uses a pre-trained ResNet50 network with additional dense layers.
- **Text Branch**: Embeds text sequences and processes them through an LSTM layer.
- **Combination**: The branches are concatenated and passed through dense layers for classification.

The final model predicts entity classes based on image and text data.

### Usage

1. **Prepare Data**: Run the following code to load, encode, and prepare data for model training.
   ```python
   file_path = r'D:\Downloads\updated_dataset_with_ocr_results.csv'
   df = load_data(file_path)
   df, num_entities = encode_entities(df)
   X_img, X_text, y = prepare_data(df)
   ```
2. **Train-Test Split**:
   ```python
   X_img_train, X_img_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
       X_img, X_text, y, test_size=0.2, random_state=42
   )
   ```
3. **Model Training**:
   ```python
   model = create_model(img_shape=X_img_train.shape[1:], text_shape=X_text_train.shape[1], num_entities=y_train.shape[1])
   model = compile_model(model)
   history = train_model(model, X_img_train, X_text_train, y_train, X_img_test, X_text_test, y_test)
   model.save('entity_extraction_model.h5')
   ```

### Evaluation

1. **Evaluate Model**:
   ```python
   y_pred_classes, y_true_classes = evaluate_model(model, X_img_test, X_text_test, y_test)
   plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
   ```
2. **Confusion Matrix**: Plots the matrix using `plot_confusion_matrix()`.

