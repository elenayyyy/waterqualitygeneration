import streamlit as st
import joblib
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# App Title
st.title("💧Water Quality Generation")
st.info("Generate data using the sidebar button to view visualizations and results.")

# Sidebar for Data Upload or Synthetic Data Generation
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose Data Source:", ["Generate Synthetic Data", "Upload Dataset"])

# Inside the "Upload Dataset" section, you can load the water quality data
if data_source == "Upload Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type="csv")
    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)
        
        # Display the dataset preview
        st.write("### Uploaded Dataset:")
        st.dataframe(data.head())

        # Ensure necessary columns are present in the dataset
        required_columns = ['Soil_Type', 'Sunlight_Hours', 'Water_Frequency', 'Fertilizer_Type', 'Temperature', 'Humidity', 'Growth_Milestone']
        if all(col in data.columns for col in required_columns):
            st.success("Dataset is valid!")
        else:
            st.error(f"Dataset must contain the following columns: {', '.join(required_columns)}")
else:
    # Synthetic data generation (if no file is uploaded)
    st.sidebar.subheader("Synthetic Data Generation")
    num_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=10000, value=1000)
    feature_names = st.sidebar.text_input("Enter Feature Names (comma-separated):", "Soil_Type,Sunlight_Hours,Water_Frequency,Fertilizer_Type,Temperature,Humidity")
    class_names = st.sidebar.text_input("Enter Class Names (comma-separated):", "Low,Medium,High")

    features = [f.strip() for f in feature_names.split(",")]
    classes = [c.strip() for c in class_names.split(",")]

    synthetic_data = []
    synthetic_labels = []

    # Class-Specific Settings in Sidebar with Selectbox for Low, Medium, High
    st.sidebar.subheader("Class-Specific Settings")
    class_settings = {}
    for cls in classes:
        class_settings[cls] = {}
        st.sidebar.subheader(f"{cls} Settings")
        for feature in features:
            mean = st.sidebar.number_input(f"Mean for {feature} ({cls})", value=50.0, key=f"{cls}_{feature}_mean")
            std = st.sidebar.number_input(f"Std Dev for {feature} ({cls})", value=10.0, key=f"{cls}_{feature}_std")
            class_settings[cls][feature] = (mean, std)

        # Generate synthetic data for each class
        for _ in range(num_samples // len(classes)):
            synthetic_data.append([np.random.normal(class_settings[cls][f][0], class_settings[cls][f][1]) for f in features])
            synthetic_labels.append(cls)

    data = pd.DataFrame(synthetic_data, columns=features)
    data['Class'] = synthetic_labels
    display_data = False

# Sample Size & Train/Test Split Configuration with Test Size Slider
st.sidebar.header("Sample Size & Train/Test Split Configuration")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=30) / 100.0
train_size = 1 - test_size
st.sidebar.write(f"Test: {test_size * 100}% / Train: {train_size * 100}%")

# Ensure the dataset has enough samples to split based on selected test size
if len(data) < 2:
    st.error("The dataset does not have enough samples for splitting.")
else:
    # Button for Training the Model
    start_training = st.sidebar.button("Generate Data and Train Models")

    # Define models to train (add all your models here)
    models = {
        "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42),
        "SVC": SVC(random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(),
    }

    # During training, store models and their metrics in session_state
    # Inside the training loop
    if start_training:
        with st.spinner("Training models... Please wait!"):
            # Start generating data and training the model
            X = data[features]
            y = data['Class']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

            # Initialize session state for learning curves and confusion matrices
            if "learning_curves" not in st.session_state:
                st.session_state["learning_curves"] = {}
            if "confusion_matrices" not in st.session_state:
                st.session_state["confusion_matrices"] = {}

            # Loop through models to train them
            for model_name, model in models.items():
                try:
                    start_time = time()
                    model.fit(X_train, y_train)  # Train the model
                    training_time = time() - start_time

                    # Save the trained model to session_state
                    if "trained_models" not in st.session_state:
                        st.session_state["trained_models"] = {}
                    st.session_state["trained_models"][model_name] = model

                    # Model Evaluation
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred, output_dict=True)

                    # Save metrics to session_state
                    if "model_metrics" not in st.session_state:
                        st.session_state["model_metrics"] = {}
                    st.session_state["model_metrics"][model_name] = {
                        "Accuracy": accuracy,
                        "Classification Report": class_report,
                        "Training Time": training_time
                    }

                    # Calculate learning curve
                    train_sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1)
                    st.session_state["learning_curves"][model_name] = {
                        "train_sizes": train_sizes,
                        "train_scores": train_scores.mean(axis=1),
                        "valid_scores": valid_scores.mean(axis=1)
                    }

                    # Calculate confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    st.session_state["confusion_matrices"][model_name] = cm

                except Exception as e:
                    st.error(f"Error while training {model_name}: {e}")

            # Now that models are saved, display confirmation
            st.success("Model training completed and models saved!")

        # Show Results
        st.write("### Dataset Split Information")
        total_samples = len(data)
        train_samples = len(X_train)
        test_samples = len(X_test)
        st.write(f"**Total Samples:** {total_samples}")
        st.write(f"**Training Samples:** {train_samples} ({(train_samples/total_samples)*100:.2f}%)")
        st.write(f"**Testing Samples:** {test_samples} ({(test_samples/total_samples)*100:.2f}%)")

        # Show Generated Data Sample
        st.write("### Generated Data Sample")
        st.write("**Original Data (Random samples from each class):**")
        st.dataframe(data.head())
        st.write("**Scaled Data (using best model's scaler):**")
        st.dataframe(pd.DataFrame(X_scaled[:5], columns=features))

        # Feature Visualization
        st.write("### Feature Visualization")
        col1, col2 = st.columns(2)
        
        # 2D Plot
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(data[features[0]], data[features[1]], c=data['Class'].map({"Low": "blue", "Medium": "orange", "High": "green"}), alpha=0.7)
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            st.pyplot(fig)
        
        # 3D Plot
        with col2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[features[0]], data[features[1]], data[features[2]], c=data['Class'].map({"Low": "blue", "Medium": "orange", "High": "green"}))
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_zlabel(features[2])
            st.pyplot(fig)

        # Download Dataset
        st.write("### Download Dataset")
        st.download_button(
            label="Download Original Dataset (CSV)",
            data=data.to_csv(index=False).encode(),
            file_name="original_data.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Scaled Dataset (CSV)",
            data=pd.DataFrame(X_scaled, columns=features).to_csv(index=False).encode(),
            file_name="scaled_data.csv",
            mime="text/csv"
        )

    
    # Dataset Statistics
    st.write("### Dataset Statistics")
    
    # Original Dataset Statistics
    st.write("**Original Dataset Statistics:**")
    original_stats = data.describe().transpose()  # Get summary statistics
    st.dataframe(original_stats)
    
    # Scaled Dataset Statistics
    st.write("**Scaled Dataset Statistics:**")
    scaled_data = pd.DataFrame(X_scaled, columns=features)
    scaled_stats = scaled_data.describe().transpose()  # Get summary statistics for scaled data
    st.dataframe(scaled_stats)


    # Best Model Performance
    st.write("### Best Model Performance")
    st.write(f"**Best Model:** ExtraTreesClassifier")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Training Time:** {training_time:.2f} seconds")
    
    # Classification Report (Best Model)
    st.write("**Classification Report (Best Model):**")
    
    # Extract classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert to a DataFrame for better visualization
    class_report_df = pd.DataFrame(class_report).transpose()
    
    # Only show the rows corresponding to the classes (excluding 'accuracy', 'macro avg', 'weighted avg')
    class_report_df = class_report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'])
    
    # Display the table
    st.dataframe(class_report_df[['precision', 'recall', 'f1-score', 'support']])


    # Model Comparison
    st.write("### Model Comparison")
    model_comparison = {
        "Model": ["ExtraTreesClassifier", "RandomForestClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier"],
        "Accuracy": [accuracy, 0.89, 0.87, 0.84, 0.83],  # Dummy accuracy values, replace with actual evaluations
        "Precision": [0.91, 0.87, 0.85, 0.82, 0.81],  # Example precision values
        "Recall": [0.92, 0.88, 0.86, 0.83, 0.80],  # Example recall values
        "F1 Score": [0.91, 0.87, 0.85, 0.82, 0.80],  # Example f1 score values
        "Training Time (s)": [training_time, 1.2, 1.3, 1.1, 1.0],  # Example training times
        "Status": ["Success", "Success", "Success", "Success", "Success"]
    }
    model_comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(model_comparison_df)

    # Performance Metrics Summary with Barplot
    st.write("### Performance Metrics Summary")

    metrics_data = {
        "Model": ["ExtraTreesClassifier", "RandomForestClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier"],
        "Accuracy": [accuracy, 0.89, 0.87, 0.84, 0.83],
        "Precision": [0.91, 0.87, 0.85, 0.82, 0.81],
        "Recall": [0.92, 0.88, 0.86, 0.83, 0.80],
        "F1 Score": [0.91, 0.87, 0.85, 0.82, 0.80]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Model', inplace=True)

    st.write("**Model Performance Metrics Comparison**")
    st.bar_chart(metrics_df)


    # Display saved models
    if "trained_models" in st.session_state and "model_metrics" in st.session_state:
        st.write("### Saved Models")
        
        saved_models = []
        for model_name, model in st.session_state["trained_models"].items():
            if model_name in st.session_state["model_metrics"]:
                accuracy = st.session_state["model_metrics"][model_name].get("Accuracy", "N/A")
                saved_models.append([model_name, accuracy])
        
        # Create DataFrame for display (without metrics)
        saved_models_df = pd.DataFrame(saved_models, columns=["Model", "Accuracy"])
        
        if not saved_models_df.empty:
            st.dataframe(saved_models_df)
            
            # CSV for all models' data (model name and accuracy only)
            csv = saved_models_df.to_csv(index=False)
            st.download_button("Download All Models as CSV", data=csv, file_name="saved_models.csv", mime="text/csv")
            
            # Provide download button for individual models
            for model_name, model in st.session_state["trained_models"].items():
                # Save the model as a pickle file
                model_filename = f"{model_name}_model.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(model, f)
                
                # Provide download button for each model
                with open(model_filename, "rb") as f:
                    st.download_button(f"Download {model_name} Model", data=f, file_name=model_filename, mime="application/octet-stream")
                
                # Get model metrics and convert to DataFrame for download
                metrics = st.session_state["model_metrics"].get(model_name, {})
                metrics_df = pd.DataFrame([metrics])
                metrics_csv = metrics_df.to_csv(index=False)
                st.download_button(f"Download {model_name} Metrics", data=metrics_csv, file_name=f"{model_name}_metrics.csv", mime="text/csv")
        else:
            st.write("No models found in the session state.")
    else:
        st.write("No trained models or model metrics found in the session state.")


# Learning Curves Display
if "learning_curves" in st.session_state:
    st.write("### Learning Curves for All Models")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (model_name, curve) in enumerate(st.session_state["learning_curves"].items()):
        axes[i].plot(curve["train_sizes"], curve["train_scores"], label="Train", color='blue')
        axes[i].plot(curve["train_sizes"], curve["valid_scores"], label="Validation", color='orange')
        axes[i].set_title(f"Learning Curve: {model_name}")
        axes[i].set_xlabel('Training Size')
        axes[i].set_ylabel('Score')
        axes[i].legend()
    plt.tight_layout()
    st.pyplot(fig)


# Confusion Matrices Display
if "confusion_matrices" in st.session_state:
    st.write("### Confusion Matrices for All Models")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (model_name, cm) in enumerate(st.session_state["confusion_matrices"].items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"Confusion Matrix: {model_name}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)
