import random
import json

# Data generation parameters
num_rows = 100
female_ratio = 0.4
income_range = (30000, 75000)
education_distribution = {
    "High School": {"weight": 0.2, "income_multiplier": 0.8},
    "Associate's": {"weight": 0.5, "income_multiplier": 1.0},
    "Master's": {"weight": 0.1, "income_multiplier": 1.2},
    "Bachelor's": {"weight": 0.2, "income_multiplier": 1.1}
}
generation_distribution = {
    "Gen Z": {"weight": 0.2},
    "Millennial": {"weight": 0.3},
    "Gen X": {"weight": 0.4},
    "Boomer": {"weight": 0.1}
}

# Generate data
data = []
for _ in range(num_rows):
    # Generate random gender
    gender = random.choices(["Male", "Female"], weights=[1 - female_ratio, female_ratio])[0]

    # Generate random education
    education = random.choices(
        list(education_distribution.keys()),
        weights=[v["weight"] for v in education_distribution.values()]
    )[0]

    # Generate random generation
    generation = random.choices(
        list(generation_distribution.keys()),
        weights=[v["weight"] for v in generation_distribution.values()]
    )[0]

    # Adjust education for female boomers
    if gender == "Female" and generation == "Boomer":
        if generation == "Boomer":
            if random.random() < 0.8:
                education = "High School"
            elif random.random() < 0.6:
                education = "Associate's"
            elif random.random() < 0.3:
                education = "Bachelor's"
            else:
                education = "Master's"
        if generation == "Gen X":
            if random.random() < 0.6:
                education = "High School"
            elif random.random() < 0.6:
                education = "Associate's"
            elif random.random() < 0.4:
                education = "Bachelor's"
            else:
                education = "Master's"
        if generation == "Millennial":
            if random.random() < 0.4:
                education = "Associate's"
            elif random.random() < 0.2:
                education = "Bachelor's"
            elif random.random() < 0.2:
                education = "High School"
            else:
                education = "Master's"

    # Generate random income
    income_multiplier = education_distribution[education]["income_multiplier"]
    income = random.uniform(income_range[0], income_range[1]) * income_multiplier

    # Adjust income for females
    if gender == "Female":
        if random.random() < 0.4:
            income *= 0.8  # Reduce income by 20%

    # Create data row
    row = {"gender": gender, "education": education, "income": round(income, 2), "generation": generation}
    data.append(row)

# Convert data to JSON
json_data = json.dumps(data, indent=4)

# Save JSON data to a file
filename = "data.json"  # Specify the desired filename
with open(filename, "w") as file:
    file.write(json_data)

print(f"JSON data saved to {filename}")
