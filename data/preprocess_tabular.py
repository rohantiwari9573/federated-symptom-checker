"""
Preprocess the Tabular Symptom dataset.
One-hot encodes symptoms, label-encodes diseases, splits into train/val/test,
and saves as PyTorch tensors.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "disease_symptom"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "tabular"

def preprocess_tabular():
    dataset_file = RAW_DIR / "dataset.csv"
    if not dataset_file.exists():
        logger.warning(f"Dataset not found at {dataset_file}. Generating synthetic symptom-disease benchmark dataset...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        # Create realistic synthetic symptom dataset (41 diseases, 132 symptoms, 4920 samples)
        diseases = [
            "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction",
            "Peptic ulcer diseae", "AIDS", "Diabetes", "Gastroenteritis", "Bronchial Asthma",
            "Hypertension", "Migraine", "Cervical spondylosis", "Paralysis (brain hemorrhage)",
            "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A",
            "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis",
            "Tuberculosis", "Common Cold", "Pneumonia", "Dimorphic hemmorhoids(piles)",
            "Heart attack", "Varicose veins", "Hypothyroidism", "Hyperthyroidism", "Hypoglycemia",
            "Osteoarthristis", "Arthritis", "(vertigo) Paroymsal  Positional Vertigo", "Acne",
            "Urinary tract infection", "Psoriasis", "Impetigo"
        ]
        symptom_list = [
            "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
            "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
            "vomiting", "burning_micturition", "spotting_urination", "fatigue", "weight_gain",
            "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
            "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
            "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
            "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes",
            "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
            "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
            "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm",
            "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
            "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
            "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
            "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
            "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
            "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
            "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck",
            "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance",
            "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
            "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching",
            "toxic_look_typhos", "depression", "irritability", "muscle_pain", "altered_sensorium",
            "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches",
            "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
            "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
            "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
            "history_of_alcohol_consumption", "blood_in_sputum", "prominent_veins_on_calf",
            "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring",
            "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
            "blister", "red_sore_around_nose", "yellow_crust_ooze"
        ] # 132 symptoms
        np.random.seed(42)
        records = []
        for disease in diseases:
            # Pick 3 to 7 characteristic symptoms for each disease
            num_syms = np.random.randint(3, 8)
            disease_syms = list(np.random.choice(symptom_list, size=num_syms, replace=False))
            for _ in range(120): # 120 samples per disease = 4920 total
                # Add slight noise (drop 1 symptom or add 1 random symptom)
                sample_syms = list(disease_syms)
                if len(sample_syms) > 2 and np.random.rand() < 0.2:
                    sample_syms.pop(np.random.randint(len(sample_syms)))
                if np.random.rand() < 0.2:
                    sample_syms.append(np.random.choice(symptom_list))
                row = {'Disease': disease}
                for idx, sym in enumerate(sample_syms[:17], 1):
                    row[f'Symptom_{idx}'] = sym
                records.append(row)
        synth_df = pd.DataFrame(records)
        synth_df.to_csv(dataset_file, index=False)
        logger.info(f"Synthetic dataset created with {len(synth_df)} records.")

    logger.info("Loading tabular dataset...")
    df = pd.read_csv(dataset_file)
    
    # Fill NaNs with empty string
    df = df.fillna('')
    
    # The first column is 'Disease', the rest are 'Symptom_1', 'Symptom_2', ...
    diseases = df['Disease'].values
    symptoms = df.drop('Disease', axis=1).values
    
    # Extract unique symptoms
    unique_symptoms = set()
    for row in symptoms:
        for symptom in row:
            if symptom.strip() != '':
                unique_symptoms.add(symptom.strip())
                
    unique_symptoms = sorted(list(unique_symptoms))
    symptom_to_idx = {sym: i for i, sym in enumerate(unique_symptoms)}
    
    logger.info(f"Found {len(unique_symptoms)} unique symptoms.")
    
    # One-hot encode symptoms
    X = np.zeros((len(df), len(unique_symptoms)), dtype=np.float32)
    for i, row in enumerate(symptoms):
        for symptom in row:
            sym = symptom.strip()
            if sym != '':
                X[i, symptom_to_idx[sym]] = 1.0
                
    # Label encode diseases
    le = LabelEncoder()
    y = le.fit_transform(diseases)
    num_classes = len(le.classes_)
    logger.info(f"Found {num_classes} unique diseases.")
    
    # Train/Val/Test Split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Save processed tensors
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    torch.save(torch.tensor(X_train), PROCESSED_DIR / "X_train.pt")
    torch.save(torch.tensor(y_train), PROCESSED_DIR / "y_train.pt")
    
    torch.save(torch.tensor(X_val), PROCESSED_DIR / "X_val.pt")
    torch.save(torch.tensor(y_val), PROCESSED_DIR / "y_val.pt")
    
    torch.save(torch.tensor(X_test), PROCESSED_DIR / "X_test.pt")
    torch.save(torch.tensor(y_test), PROCESSED_DIR / "y_test.pt")
    
    logger.info("Saved processed tabular dataset.")
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

if __name__ == "__main__":
    preprocess_tabular()
