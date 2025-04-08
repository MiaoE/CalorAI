import pandas as pd
import json
import re
import csv


database_csv = 'calories_database.csv'
database_json = 'calories_database.json'

data_labels_csv = 'data_labels_raw.csv'
data_labels_json = 'data.json'

def get_calories_per_gram(serving, calories):
        """Gets the unit calorie content given a total calorie content and its serving"""
        weight_match = re.search(r"(\d+)\s*g", serving)
        cal_match = re.search(r"(\d+)", str(calories))
        
        if weight_match and cal_match:
            weight = float(weight_match.group(1))
            cal = float(cal_match.group(1))
            return round(cal / weight, 5) if weight > 0 else None
        return None

def parse_database_csv_to_json(database_csv, database_json):
    """Converts the database CSV file to its designated JSON file specified by path"""
    df = pd.read_csv(database_csv)
    
    calories_per_gram_dict = {
        row["Food"].strip(): float(get_calories_per_gram(row["Serving"], row["Calories"]))
        for _, row in df.iterrows()
        if get_calories_per_gram(row["Serving"], row["Calories"]) is not None

    }
    
    json_output = json.dumps(calories_per_gram_dict, indent=4)
    
    with open(database_json, "w") as f:
        f.write(json_output)
    
    return json_output


def get_total_calories(database_json, food, weight):
    """Returns the calorie content of a specific food and its weight"""
    with open(database_json, "r") as f:
        calories_data = json.load(f)
    
    food = food.strip()

    if food in calories_data and isinstance(calories_data[food], float):
        return round(calories_data[food] * float(weight), 2)
    else:
        return f"Calorie data for {food} not found."


def csv_to_json(data_labels_csv, data_labels_json, database_json):
    """Converts the custom data CSV file to a JSON file specified by path"""
    data = []

    # Read CSV and process data
    with open(data_labels_csv, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            figure_name = row["Figure Name"].strip()
            food_items_raw = row["Food Items "].strip() 
            portion_sizes_raw = row["Portion Size (g)"].strip()

            # Handle cases where there's only one food item
            food_types = [food.strip() for food in food_items_raw.split(",")]
            portions = [portion.strip() for portion in portion_sizes_raw.split(",")]

            # Ensure food types and portions are mapped correctly
            if len(food_types) != len(portions):
                print(f"Warning: Mismatch in food types and portions for {figure_name}. Data might be incorrect.")

            # Calculate calorie values
            calories = [get_total_calories(database_json, food, portion) for food, portion in zip(food_types, portions)]

            formatted_entry = {
                "name": figure_name,
                "food type": food_types,
                "portion": portions,
                "calorie": calories  # Store calculated calories
            }
            data.append(formatted_entry)

    with open(data_labels_json, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"CSV converted to JSON successfully: {data_labels_json}")


csv_to_json(data_labels_csv, data_labels_json, database_json)
