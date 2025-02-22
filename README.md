# Parkinson's Disease Early Diagnosis and Classification using IoT Smartwatch Data

## Project Overview
This project applies deep learning techniques to raw smartwatch sensor data from the Parkinson's Disease Smartwatch (PADS) dataset to enable early diagnosis and prediction of Parkinson's Disease (PD). The system leverages IoT-based wearable devices, cloud storage, and machine learning models to analyze motion patterns and provide insights for neurologists and patients.
The primary objective is to develop predictive models for classifying PD and forecasting movement variability using CNN-LSTM and LSTM architectures.

## Dataset Information
- **Source:** PhysioNet - Parkinson’s Disease Smartwatch (PADS) Dataset ([Link](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/))
- **Observations:** 5,159 movement task recordings from 469 participants
- **Sensors:** Accelerometer (X, Y, Z), Gyroscope (X, Y, Z) sampled at 100Hz
- **Classes:** Parkinson’s Patients (PD), Differential Diagnoses (DD), Healthy Controls (HC)

## Project Workflow
1. **Data Processing**
   - Parse patient metadata and session details from JSON files.

   - Load and merge sensor time-series data.

   - Assign demographic details to each observation.

   - Save the processed dataset for further analysis.
   
2. **Exploratory Data Analysis (EDA) and Cleaning**

   - Identify missing values and outliers.

   - Normalize numerical features.

   - Convert categorical features (Gender, Handedness, Condition) to binary representations.

   - Save cleaned data for model training.
   
3. **Deep Learning Models**
   - **CNN-LSTM Model** (for PD classification):
     - CNN extracts spatial features from sensor data.
     - LSTM captures temporal dependencies.
     - Binary classification of PD vs. Healthy patients.
   - **LSTM Model** (for movement prediction):
     - Predicts future sensor readings.
     - Helps understand movement variability in PD patients.

4. **Model Training and Evaluation**
   - Train CNN-LSTM and LSTM models using time-sequenced patient data.
   - Evaluate models using accuracy, loss, and mean absolute error (MAE).
   - Generate confusion matrices and classification reports.
   - Save results for visualization in Tableau.
  
5. **Visualization**
   - Compute loss, accuracy, precision, recall, and F1-score.
   - Generate confusion matrices and feature importance plots.
   - Export results to Tableau for data visualization.
   
6. **Deployment**
   - Save the trained model for inference.

   

## Tools & Libraries Used

This project utilizes the following tools and libraries:

- **Programming Language:** `Python`

- **Data Processing:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

- **Visualization:** `matplotlib`, `seaborn`, `Tableau`

- **Machine Learning:** `scikit-learn`, `TensorFlow`, `PyTorch`, `Keras`



## Installation & Requirements
### **Prerequisites**
Ensure you have Python and the required dependencies installed:
```bash
pip install numpy pandas matplotlib seaborn tensorflow pytorch scikit-learn
```

### **Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/Fatimat01/PD-Detection.git
   cd notebooks/
   ```
2. Process the raw data:
   ```bash
   jupyter notebook data-processing.ipynb
   ```
3. EDA & Preprocess Data
   ```bash
   jupyter notebook clean-data.ipynb
   ```
4. Train the deep learning model (CNN-LSTM):
   ```bash
   jupyter notebook cnn-lstm.ipynb
   lstm.ipynb
   ```


## Project Structure
```
project/
├── notebooks/          # Jupyter Notebooks for processing, training, and evaluation
├── models/             # Saved trained models
├── README.md           # Project documentation
```

## Future Enhancements
- Implement real-time data streaming for continuous monitoring.
- Optimize deep learning models for edge deployment.
- Integrate multimodal data (questionnaire responses) for enhanced prediction.
- Deploy the model as a cloud-based inference API.

## References
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
- Varghese, J., Brenner, A., Plagwitz, L., van Alen, C., Fujarski, M., & Warnecke, T. (2024). PADS - Parkinsons Disease Smartwatch dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/m0w9-zx22.
Varghese, J., Brenner, A., Fujarski, M. et al. Machine Learning in the Parkinson’s disease smartwatch (PADS) dataset. npj Parkinsons Dis. 10, 9 (2024). https://doi.org/10.1038/s41531-023-00625-7


## Contact
For questions or collaboration opportunities, feel free to reach out!

