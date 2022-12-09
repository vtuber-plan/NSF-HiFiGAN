import argparse
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="./filelists/48k_audio_filelist.txt", help='filelist path')
    parser.add_argument('-o', '--output', type=str, default="./filelists", help='File list output path')
    args = parser.parse_args()

    random.seed(1234)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = sorted(lines)
    random.shuffle(lines)

    origin_filename = os.path.basename(args.input)
    data_len = len(lines)

    valid_num = int(data_len * 0.001)
    test_num = int(data_len * 0.001)

    with open(os.path.join(args.output, origin_filename.replace(".txt", "_train.txt")), "w", encoding="utf-8") as f:
        f.writelines(lines[:-valid_num-test_num])
    
    with open(os.path.join(args.output, origin_filename.replace(".txt", "_valid.txt")), "w", encoding="utf-8") as f:
        f.writelines(lines[-valid_num-test_num:-test_num])
    
    with open(os.path.join(args.output, origin_filename.replace(".txt", "_test.txt")), "w", encoding="utf-8") as f:
        f.writelines(lines[-test_num:])