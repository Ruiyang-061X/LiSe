'''
Ruiyang Zhang, ruiyang.061x@gmail.com, 2024.7.11

Integrate image-based pseudo labels into LiDAR-based pseudo labels.
Consider distance-aware fusion.
'''

import argparse
import glob
from tqdm import tqdm


def integration_with_2D_scenes(args):
    B_LiDAR_file_paths = glob.glob(args.B_LiDAR_dir_path)
    for B_LiDAR_file_path in tqdm(B_LiDAR_file_paths):
        B_final_file = open(args.B_final_dir_path + B_LiDAR_file_path.split('/')[-1], 'w')
        # Add all LiDAR-based pseudo labels
        B_LiDAR = open(B_LiDAR_file_path, 'r').readlines()
        B_final_file.writelines(B_LiDAR)
        B_final_file.writelines('\n')
        # Image-based pseudo labels, distance-aware fusion
        B_img = open(args.B_img_dir_path + B_LiDAR_file_path.split('/')[-1], 'r').readlines()
        for b_i in B_img:
            d_b_i = float(b_i.split(' ')[-2])
            if d_b_i >= args.d_min:
                B_final_file.writelines(b_i)

def main():
    parser = argparse.ArgumentParser(description="Integration with 2D scenes.")
    parser.add_argument("--B_LiDAR_dir_path", type=str, required=True, help="Path to image-based pseudo labels.")
    parser.add_argument("--B_img_dir_path", type=str, required=True, help="Path to LiDAR-based pseudo labels.")
    parser.add_argument("--B_final_dir_path", type=str, required=True, help="Path to save final intergrated labels.")
    parser.add_argument("--d_min", type=float, default=10, help="Minimium distance of image-based pseudo labels.")
    args = parser.parse_args()
    integration_with_2D_scenes(args)

if __name__ == "__main__":
    main()
