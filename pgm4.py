import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Define fuzzy variables
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')  # 0 to 40 °C
heating_power = ctrl.Consequent(np.arange(0, 101, 1), 'heating_power')  # 0% to 100%

# 2. Define membership functions for temperature
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 10])
temperature['warm'] = fuzz.trimf(temperature.universe, [10, 15, 22])
temperature['hot'] = fuzz.trimf(temperature.universe, [20, 30, 40])

# 3. Define membership functions for heating power
heating_power['low'] = fuzz.trimf(heating_power.universe, [0, 0, 30])
heating_power['medium'] = fuzz.trimf(heating_power.universe, [25, 30, 55])
heating_power['high'] = fuzz.trimf(heating_power.universe, [50, 70, 90])

# 4. Define fuzzy rules
rule1 = ctrl.Rule(temperature['cold'], heating_power['high'])
rule2 = ctrl.Rule(temperature['warm'], heating_power['medium'])
rule3 = ctrl.Rule(temperature['hot'], heating_power['low'])

# 5. Create control system
heating_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
heating_simulation = ctrl.ControlSystemSimulation(heating_ctrl)

# 6. Input current temperature
current_temp = float(input("Enter current temperature (°C): "))
if current_temp < temperature.universe[0] or current_temp > temperature.universe[-1]:
    print(f"Temperature {current_temp}°C is out of range ({temperature.universe[0]}–{temperature.universe[-1]}°C). Please enter a value within range.")
else:
    heating_simulation.input['temperature'] = current_temp
    # 7. Compute output
    heating_simulation.compute()
    # 8. Output result
    print(f"Recommended heating power: {heating_simulation.output['heating_power']:.2f}%")
