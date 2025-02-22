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
   - Extract raw data from JSON and `.txt` files.
   - Preprocess missing values, synchronize timestamps, and normalize data.
   
2. **Feature Engineering**
   - Normalize and scale time-series data for deep learning models.
   
3. **Deep Learning Model**
   - Implement a CNN-LSTM hybrid model to classify PD, and HC.
   - Train, validate, and fine-tune the model.
   
4. **Time-Series Prediction Model**
   - Develop LSTM-based sequence models to predict future movement variability, and detect anomaly for early PD detection

   
5. **Evaluation and Visualization**
   - Compute accuracy, precision, recall, and F1-score.
   - Generate confusion matrices and feature importance plots.
   - Export results to Tableau for data visualization.
   
6. **Deployment**
   - Save the trained model for inference.

   

## Tools & Libraries Used

This project utilizes the following tools and libraries:

- **Programming Language:** Python

- **Data Processing:** pandas, numpy, scipy

- **Visualization:** matplotlib, seaborn, Tableau

- **Machine Learning:** scikit-learn, TensorFlow, PyTorch, Keras





## Installation & Requirements
### **Prerequisites**
Ensure you have Python and the required dependencies installed:
```bash
pip install numpy pandas matplotlib seaborn tensorflow pytorch scikit-learn tsfresh tslearn flask fastapi
```

### **Run the Project**
1. Clone the repository:
   ```bash
   git clone url
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
├── data/               # Raw and processed dataset
├── notebooks/          # Jupyter Notebooks for processing, training, and evaluation
├── models/             # Saved trained models
├── README.md           # Project documentation
```

## Future Enhancements
- Implement real-time data streaming for continuous monitoring.
- Optimize deep learning models for edge deployment.
- Integrate multimodal data (questionnaire responses) for enhanced prediction.

## References
- Varghese, J., Brenner, A., Fujarski, M. et al. (2024). Machine Learning in the Parkinson’s Disease Smartwatch (PADS) dataset. *npj Parkinson's Disease, 10*(9). https://doi.org/10.1038/s41531-023-00625-7
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation, 101*(23), e215–e220.

## Contact
For questions or collaboration opportunities, feel free to reach out!

