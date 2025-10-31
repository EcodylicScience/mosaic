import json
import os
import numpy as np 
import argparse

'''
Script for converting CalMS21 .json files into .npy files.
The .npy files have the same dictionary layout, except the entries are
numpy arrays instead of lists.
If treba features are not appended, the final dictionary 'keypoint' entries will have shape:
sequence_length x 2 x 2 x 7.
If treba features are appended, the final dictionary 'features' entries will have shape:
sequence_length x 60 (2x2x7 + 32).
'''

def convert_to_array(dictionary, feature_dictionary = None):
    # Convert dictionary values (lists) to numpy arrays, until depth 3.
    # If feature dictionary is not None, also concatenate the dictionary values.
    converted = {}

    # First key is the group name for the sequences
    for groupname in dictionary.keys():

        converted[groupname] = {}
        # Next key is the sequence id
        for sequence_id in dictionary[groupname].keys():

            converted[groupname][sequence_id] = {}

            # If not adding features, add keypoints, scores, and annotations & metadata (if available)
            if feature_dictionary is None:
                converted[groupname][sequence_id]['keypoints'] = np.array(dictionary[groupname][sequence_id]['keypoints'])
            else:
                keypoints = np.array(dictionary[groupname][sequence_id]['keypoints'])
                converted[groupname][sequence_id]['features'] = np.concatenate([keypoints.reshape(keypoints.shape[0], -1),
                                                feature_dictionary[groupname][sequence_id]['features']], axis = -1)

            converted[groupname][sequence_id]['scores'] = np.array(dictionary[groupname][sequence_id]['scores'])         
                
            if 'annotations' in dictionary[groupname][sequence_id].keys():
                converted[groupname][sequence_id]['annotations'] = np.array(dictionary[groupname][sequence_id]['annotations'])                         

            if 'metadata' in dictionary[groupname][sequence_id].keys():
                converted[groupname][sequence_id]['metadata'] = dictionary[groupname][sequence_id]['metadata']                  

    return converted


def json_save_to_npy(input_name, output_name, feature_name = None):
    with open(input_name, 'r') as fp:
        input_data = json.load(fp)

    if feature_name is not None:
        with open(feature_name, 'r') as fp:
            features_data = json.load(fp)

        input_data = convert_to_array(input_data, features_data)
    else:
        input_data = convert_to_array(input_data)

    print("Saving " + output_name)
    np.save(output_name, input_data, allow_pickle=True)    


def convert_all_calms21(args):

    calms21_files = []

    # find all files beginning with calms21 in the input dictionary and ending with .json
    for root, dirs, files in os.walk(args.input_directory):
       
        for name in files:
          if name.startswith('calms21_') and name.endswith('.json'):
            calms21_files.append(os.path.join(root, name))

    for single_file in calms21_files:
        if not args.parse_treba:
            # Parse keypoints only.
            file_name = single_file.split('/')[-1].split('.')[0]
            npy_output_name = os.path.join(args.output_directory, file_name)
            json_save_to_npy(single_file, npy_output_name)

        else:
            # Parse keypoints and concatenate with treba features.
            file_name = single_file.split('/')[-1].split('.')[0]
            npy_output_name = os.path.join(args.output_directory, file_name + '_features')

            current_root = single_file.rsplit('/', 1)[0]
            treba_feature_name = os.path.join(current_root, 'taskprog_features' + file_name[7:] + '.json')

            json_save_to_npy(single_file, npy_output_name, treba_feature_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, required = True, 
    	help='Directory to CalMS21 json files')
    parser.add_argument('--output_directory', type=str, default = 'data', required = False, 
    	help='Directory to output npy files')    
    parser.add_argument('--parse_treba', action="store_true",
                        help='Whether or not to include treba features in the npy files')

    parsed_args = parser.parse_args()

    convert_all_calms21(parsed_args)
