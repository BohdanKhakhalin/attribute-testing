import os
from pathlib import Path

import pandas as pd
import requests
import argparse
from datetime import datetime
import time
import json

# set up parser and define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', help='path to input csv file', required=True)
parser.add_argument('--platform_url', help='platform url', required=True)
parser.add_argument('--odin_id', help='bot odin id', required=True)
parser.add_argument('--bot_name', help='bot name', required=False)
parser.add_argument('--custom_delay', type=int, help='custom delay in seconds', required=False, default=1)


# create function that gets value of custom delay from user
def get_custom_delay():
    user_input = input('Provide delay in seconds: ').strip()
    return int(user_input) if user_input.isdigit() else 0


# store values provided for command-line arguments
args = parser.parse_args()

# ask for argument values if not provided in command line
input_file = args.input_file if args.input_file else input('Provide path to input csv file: ')
platform_url = args.platform_url if args.platform_url else input('Provide platform url: ')
odin_id = args.odin_id if args.odin_id else input('Provide bot odin id: ')
bot_name = args.bot_name if args.bot_name else input('Provide bot name: ')
custom_delay = args.custom_delay if args.custom_delay is not None else get_custom_delay()

# read input.csv file into DataFrame
test_data = pd.read_csv(input_file)

columns_list = test_data.columns.tolist()

# define entity syntax elements
ENTITY_DELIMITER = '-||-'
ATTRIBUTE_VALUES_DELIMITER = '=='
VALUES_DELIMITER = '-|-'
ORIGINAL_RESOLVED_VALUE_DELIMITER = '=>'
NO_ENTITIES_PLACEHOLDER = '--'


# create function that performs Recognize intent request
def recognize_intent(platform_url, odin_id, user_phrase):
    request_url = platform_url + '/bot/' + odin_id + '/intent/recognize'
    request_body = {"query": user_phrase}
    response = requests.post(request_url, json=request_body)
    return response.text, response.status_code


# create function that stores results to output file
def save_result(bot_name):
    output_filename = 'entity_accuracy_results_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
    if bot_name:
        output_filename = bot_name + '_' + output_filename
    dir_path = 'test_results'
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(dir_path, output_filename)
    test_data.to_csv(file_path, index=False, na_rep="n/a")


# create function to check if response is a valid json
def is_json(value):
    try:
        json.loads(value)
        return True
    except ValueError:
        return False


# create function that parses expected entities field value from the input file into separate entities
def split_entities(entities):
    if ENTITY_DELIMITER in entities:
        entity_list = [s.strip() for s in entities.split(ENTITY_DELIMITER)]
    else:
        entity_list = [entities]
    return entity_list


# create function that parses expected data for each entity from the input file into list of dictionaries
def get_entity_objects_list(entities):
    try:
        entity_list = split_entities(entities)
        if len(entity_list) == 1 and entity_list[0] == NO_ENTITIES_PLACEHOLDER:
            return []
        entity_objects_list = []
        for entity in entity_list:
            attribute_name, entity_values = entity.split(ATTRIBUTE_VALUES_DELIMITER)
            value_pairs = entity_values.split(VALUES_DELIMITER)
            values = []
            for pair in value_pairs:
                original_value, resolved_value = pair.split(ORIGINAL_RESOLVED_VALUE_DELIMITER)
                value_object = {
                    'original_value': original_value,
                    'resolved_value': resolved_value
                }
                values.append(value_object)

            entity_objects_list.append({
                'attribute_name': attribute_name,
                'values': values
            })

        return entity_objects_list
    except ValueError:
        return None


# create function that gets list of recognized entities from the response
def get_response_entity_objects_list(response):
    entities = []
    if is_json(response):
        entities = json.loads(response)['entities']
        if not entities:
            return []
    response_entity_objects_list = []
    for entity in entities:
        attribute_name = entity["attribute_name"]
        values = []
        for value_pair in entity['values']:
            original_value = value_pair["original_value"]
            resolved_value = value_pair["resolved_value"]
            value_object = {
                'original_value': original_value,
                'resolved_value': resolved_value
            }
            values.append(value_object)
        response_entity_objects_list.append({
            'attribute_name': attribute_name,
            'values': values
        })

    return response_entity_objects_list


# create function that transforms list of recognized entities to string (for test results file)
def save_recognized_entities(recognized_entities):
    if not recognized_entities:
        return NO_ENTITIES_PLACEHOLDER
    entities = []
    for entity in recognized_entities:
        attribute_name = entity['attribute_name']
        values = entity['values']
        value_pairs = []
        for value_object in values:
            original_value = value_object['original_value']
            resolved_value = value_object['resolved_value']
            value_pairs.append(f"{original_value}=>{resolved_value}")
        entity_values = VALUES_DELIMITER.join(value_pairs)
        entity = f"{attribute_name}=={entity_values}"
        entities.append(entity)
    return ENTITY_DELIMITER.join(entities)


# create function that gets resolved value from each object in the values list
def get_resolved_value(value_pair):
    resolved_value = value_pair.get("resolved_value")
    return resolved_value


# sort resolved values inside each entity alphabetically
def sort_resolved_values(entity_list):
    for entity in entity_list:
        if "values" in entity:
            entity["values"] = sorted(entity["values"], key=get_resolved_value)
    return entity_list


# create function that compares list of expected entities with list of actual entities
def compare_entity_lists(entity_list, response_entity_list):
    if not isinstance(response_entity_list, list):
        return False
    sorted_values_entity_list = sort_resolved_values(entity_list)
    sorted_values_response_entity_list = sort_resolved_values(response_entity_list)
    key_to_map = 'attribute_name'
    left_set = {item[key_to_map]: item for item in sorted_values_entity_list}
    right_set = {item[key_to_map]: item for item in sorted_values_response_entity_list}
    if left_set == right_set:
        return True
    return False


# iterate over each row in the DataFrame
for index, row in test_data.iterrows():
    print(row['user_phrase'], row['intent_name'])
    # make recognize intent request for each user phrase and store recognized intent
    response, status_code = recognize_intent(platform_url, odin_id, row['user_phrase'])
    recognized_intent_name = json.loads(response)['name'].replace("FAQ#&name=", "") if status_code == 200 else response
    if 'entities' in columns_list and pd.notna(row['entities']):
        print(row['entities'])
        # get expected entities from file
        expected_entities = get_entity_objects_list(row['entities'])
        # get actual entities from response
        actual_entities = get_response_entity_objects_list(response)
        # compare expected and actual entities
        if expected_entities is not None:
            entity_comparison_result = compare_entity_lists(expected_entities,
                                                            actual_entities)
        # if intent name in the input.csv and in response is the same and entities also match, mark as correct
            test_data.loc[index, 'correct'] = row[
                                              'intent_name'] == recognized_intent_name and entity_comparison_result is True
        else:
            # display custom error in "correct" column in case expected entities were not parsed
            test_data.loc[index, 'correct'] = 'Invalid expected entity format'
        # add actual entities to result file
        test_data.loc[index, 'recognized_entities'] = save_recognized_entities(
            actual_entities) if status_code == 200 else response
    else:
        # if intent name in the input.csv and in response is the same, mark as correct
        test_data.loc[index, 'correct'] = row['intent_name'] == recognized_intent_name
    # add recognized intent name to result file
    test_data.loc[index, 'recognized_intent_name'] = recognized_intent_name
    # set custom delay between requests to avoid server overload
    time.sleep(custom_delay)

save_result(bot_name)
# replace custom error with "False" before calculating accuracy %
test_data['correct'] = test_data['correct'].replace('Invalid expected entity format', False)
# print accuracy in %
print('Accuracy: ', round(test_data['correct'].mean() * 100, 2), '%')

# how to run script
# python3 <path to entity_accuracy_check.py> --input_file <path to input csv file>  --platform_url <url> --odin_id <bot odin id> --custom_delay <delay in sec>
