import glob
import os


def scan_dict(path):
    dirs = os.listdir(path)
    per_files = {}
    for idx in range(len(dirs)):
        f_name = dirs[idx]
        per_files[f_name] = glob.glob(path + '/' + f_name + '/*.txt')

    return per_files


if __name__ == '__main__':
    ff = scan_dict('H:\ATL03_DATASET\BCGD_2016_REGION\Converted')
    d = 0

