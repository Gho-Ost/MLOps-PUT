import json
import urllib.request
import sys
import os

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
CACHE_FILE = "imagenet_labels.json"

def get_labels():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
            
    print("Downloading ImageNet labels...")
    with urllib.request.urlopen(LABELS_URL) as response:
        labels = json.loads(response.read())
        
    with open(CACHE_FILE, 'w') as f:
        json.dump(labels, f)
        
    return labels

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_class_name.py <result_json_file>")
        sys.exit(1)
        
    result_file = sys.argv[1]
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            
        class_id = data.get('class_id')
        if class_id is None:
            print("Error: 'class_id' not found in JSON file.")
            sys.exit(1)
            
        labels = get_labels()
        
        if 0 <= class_id < len(labels):
            print(f"Class ID: {class_id}")
            print(f"Class Name: {labels[class_id]}")
        else:
             print(f"Error: Class ID {class_id} is out of range (0-{len(labels)-1}).")
             
    except FileNotFoundError:
        print(f"Error: File '{result_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{result_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
