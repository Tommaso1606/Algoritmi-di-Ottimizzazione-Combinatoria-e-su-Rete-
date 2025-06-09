from joblib import load
from typing import Dict

FILE = "firstSolutionGreedy.joblib"

def main():
    
    dictionary: Dict = load(FILE)
    
    print(dictionary.get("Distance"))
    
    return

if __name__ == "__main__":
    main()






