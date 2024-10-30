import weka.core.jvm as jvm
from weka.core.converters import load_any_file, save_any_file
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation
import pandas as pd
import tempfile
import os
from compile_data import *
import argparse

# Read CSV file
csv_file = '/Users/archit/Documents/Watch_hand_wash_classify/balanced_features.csv'

# Save ARFF file
arff_file = '/Users/archit/Documents/Watch_hand_wash_classify/arff_data/balanced_features.arff'

class MyWekaUtils:

    def __init__(self, args=None):
        self.args = args
    
    @staticmethod
    def classify(option: int) -> float:
        # Start the JVM if not already running
        if not jvm.started:
            jvm.start()
        
        # Load the ARFF data into Weka Instances
        data = load_any_file(args.arff_path, class_index="last")
        # data.class_is_last()
        
        # Select classifier based on the option
        if option == 1:
            classifier = Classifier(classname="weka.classifiers.trees.J48")  # Decision Tree
        elif option == 2:
            classifier = Classifier(classname="weka.classifiers.trees.RandomForest")
        elif option == 3:
            classifier = Classifier(classname="weka.classifiers.functions.SMO")  # SVM
        else:
            return -1

        # Build classifier
        classifier.build_classifier(data)

        # Cross-validate the model
        evaluation = Evaluation(data)
        evaluation.crossvalidate_model(classifier, data, 10, Random(1))

        return evaluation.percent_correct

    @staticmethod
    def read_csv() -> pd.DataFrame:
        # Read CSV file into a DataFrame
        data = pd.read_csv(args.csv_path)
        if data.empty:
            print("No data found")
            return None
        return data

    @staticmethod
    def csv_to_arff(class_column: str = None) -> str:
        # Save DataFrame as a temporary CSV file
        df = pd.read_csv(args.csv_path, names = ['mean_x', 'std_x', 'mean_y', 'std_y', 'mean_z', 'std_z', 'Activity'])
        # Load the CSV as Weka Instances
        instances = load_any_file(args.csv_path)
        
        # If class_column is provided, set it as the class attribute
        if class_column:
            class_index = df.columns.get_loc(class_column)
            instances.class_index = class_index
        else:
            instances.class_index = -1  # No class attribute set; all columns included
        
        # Save Instances as ARFF to a temporary file
        save_any_file(instances, args.arff_path)
        print(f"ARFF file '{args.arff_path}' created successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand wash classification using Weka')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file', default=csv_file)
    parser.add_argument('--arff_path', type=str, help='Path to the ARFF file', default=arff_file)
    parser.add_argument('--stride', type=int, help='Stride value', default=1)
    parser.add_argument('--window_size', type=int, help='Window size', default=1000)
    args = parser.parse_args()

    # Start the JVM
    if not jvm.started:
        jvm.start()


    weka_obj = MyWekaUtils(args)
    generate_data(args.stride, args.window_size)
    args.csv_path = f'features_stride_{args.stride}_window_size_{args.window_size}.csv'
    args.arff_path = f'./arff_data/features_stride_{args.stride}_window_size_{args.window_size}.arff'

    df = weka_obj.read_csv()
    weka_obj.csv_to_arff(class_column='Activity')
    accuracy = weka_obj.classify(option= 1)
    print(f"Accuracy: {accuracy:.2f}%")

    jvm.stop()