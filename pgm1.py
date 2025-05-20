print("=== Welcome to the Car Diagnostic System ===")
print("Please answer the following questions by typing 1 for 'Yes' and 2 for 'No'.")

def ask(question):
    while True:
        print(question)
        print("1. Yes")
        print("2. No")
        choice = input("Your choice: ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("Please enter 1 for Yes or 2 for No.\n")

# Start diagnosis
if not ask("Does the car start?"):
    if ask("Are the lights dim?"):
        print("\nDiagnosis: Battery is dead")
    else:
        if ask("Does the radio work?") and ask("Are the lights bright?"):
            print("\nDiagnosis: Starter motor issue")
        else:
            if ask("Does the engine crank?"):
                if ask("Do you smell fuel?"):
                    print("\nDiagnosis: Engine is flooded")
                elif ask("Do you NOT smell fuel?"):
                    print("\nDiagnosis: Fuel pump failure")
                else:
                    print("\nDiagnosis: Could not determine the issue")
            else:
                print("\nDiagnosis: Could not determine the issue")
else:
    if ask("Does the car stall frequently?") and ask("Is the engine idle rough?"):
        print("\nDiagnosis: Spark plug problem")
    else:
        print("\nThe car seems to be working fine, or the issue is unknown.")
