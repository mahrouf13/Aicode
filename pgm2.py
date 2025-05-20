
# === Super Simple Disease Diagnosis System with Safe Input ===
# Step 1: Store diseases and their symptoms
knowledge_graph = {
    "Common Cold": ["cough", "sore throat", "runny nose", "sneezing"],
    "Flu": ["fever", "cough", "body ache", "chills"],
    "COVID-19": ["fever", "cough", "loss of taste", "difficulty breathing"],
    "Malaria": ["fever", "chills", "headache", "sweating"],
    "Allergy": ["sneezing", "runny nose", "itchy eyes"]
}
# Step 2: Create a list of all symptoms (no duplicates)
all_symptoms = [
    "cough", "sore throat", "runny nose", "sneezing", "fever", "body ache",
    "chills", "loss of taste", "difficulty breathing", "headache", "sweating", "itchy eyes"
]
print("=== Welcome to the Simple Disease Diagnosis System ===")
print("Select the symptoms you are experiencing:")
# Step 3: Show symptoms with numbers
for i in range(len(all_symptoms)):
    print(str(i + 1) + ". " + all_symptoms[i])
# Step 4: Get input from user
print("\nEnter the numbers of your symptoms separated by commas (e.g., 1,3,5):")
user_input = input("Your symptoms: ")
# Step 5: Convert and validate input
selected_symptoms = []
numbers = user_input.split(",")
for n in numbers:
    n = n.strip()
    if n.isdigit():
        index = int(n)
        if 1 <= index <= len(all_symptoms):
            symptom = all_symptoms[index - 1]
            if symptom not in selected_symptoms:
                selected_symptoms.append(symptom)
# Step 6: Show result or error
if len(selected_symptoms) == 0:
    print("\nNo valid symptoms selected. Please enter correct numbers only.")
else:
    print("\nChecking possible diseases...\n")
    for disease in knowledge_graph:
        disease_symptoms = knowledge_graph[disease]
        match_count = 0
        for symptom in selected_symptoms:
            if symptom in disease_symptoms:
                match_count = match_count + 1
        if match_count > 0:
            print(disease, " --> matched ", match_count, "symptoms.")
