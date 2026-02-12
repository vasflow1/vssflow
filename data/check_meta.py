import argparse
import csv
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="一个简单的命令行参数示例")
    
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    cav_path = args.path

    c = 0
    with open(cav_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='|')
        for row in tqdm(csv_reader):
            for path in row:
                if '/' not in path: # not a path
                    continue
                if not os.path.exists(path):
                    c += 1
                    print(path)

    print(f"Missing Items {c}")

if __name__ == '__main__':
    main()