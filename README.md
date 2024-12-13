# planitorium-ml
Machine Learning for Planitorium

## Dataset
https://www.kaggle.com/datasets/mdfikri/planitorium-dataset

# Machine Learning Pipeline for Plant Disease Detection

This repository contains a deep learning pipeline designed for detecting plant diseases, focusing specifically on maize (corn) leaves. The solution includes data preprocessing, model training, evaluation, and deployment using TensorFlow.js.

## Features
- **Image Data Preprocessing**: Using `ImageDataGenerator` for data augmentation.
- **Model Training**: A custom Convolutional Neural Network (CNN) built with TensorFlow's Sequential API.
- **Callbacks**: Includes `EarlyStopping` and `LearningRateScheduler` to optimize training.
- **Evaluation**:
  - Confusion Matrix
  - Classification Report
  - ROC AUC Analysis
- **Deployment**: Conversion of trained model to TensorFlow.js for web-based applications.

## Requirements
To run the code, you need the following dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Code Structure

1. **Data Preprocessing**
   - Utilizes `ImageDataGenerator` for real-time data augmentation.
   - Splits the dataset into training, validation, and testing subsets.

2. **Model Architecture**
   - Custom CNN with layers like `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense`.
   - Optimized with Adam optimizer and a learning rate scheduler.

3. **Training**
   - Incorporates `EarlyStopping` to avoid overfitting.
   - Visualizes training and validation loss/accuracy.

4. **Evaluation**
   - Generates confusion matrix and classification report.
   - Plots ROC curves for multi-class classification.

5. **Model Conversion**
   - Converts the trained TensorFlow model to TensorFlow.js format for web deployment.

## Usage

1. **Data Preparation**
   Place your dataset in a folder structure as follows:
   ```plaintext
   dataset/
     Antraknose/
     Karat Daun Jagung/
     ....
   ```

2. **Run Training**
   Execute the script to train the CNN model:
   ```bash
   model.fit()
   ```

3. **Evaluate Model**
   The evaluation results, including confusion matrix and ROC AUC, will be saved in the output directory.

4. **Convert to TFJS**
   Convert the trained model to TensorFlow.js format:
   ```bash
   tensorflowjs_converter --input_format=tf_saved_model <saved_model_dir> <output_dir>
   ```

5. **Deploy on Web**
   Use the exported TensorFlow.js model in your web application.

## Example Visualizations
- **Training Accuracy and Loss**:
  Displays graphs of training and validation accuracy/loss over epochs.

- **Confusion Matrix**:
  Visualizes model performance on the test set.

- **ROC Curves**:
  Demonstrates the model's ability to differentiate between classes.

## Future Improvements
- Expand disease detection to include other crops.
- Improve model accuracy with more sophisticated architectures.
- Add features like personalized crop reminders and an integrated marketplace for farmers.

## License
This project is licensed under the MIT License. Feel free to use and modify the code for your needs.

## Acknowledgments
- TensorFlow for the deep learning framework.
- Open datasets for providing labeled plant disease images.

---
If you encounter any issues or have suggestions for improvement, feel free to raise an issue or contribute to the repository!
