import os
import glob
from sklearn.model_selection import train_test_split

def find_nii_files(root_dir):
    ncct_files = []
    mask_files = []
    
    # Traverse all batches (batch_1, batch_2, etc.)
    for batch in os.listdir(root_dir):
        batch_path = os.path.join(root_dir, batch)

        # Ensure it's a valid directory
        if not os.path.isdir(batch_path):
            continue
        
        # Search inside raw_data for NCCT scans
        raw_data_path = os.path.join(batch_path, "raw_data")
        if os.path.exists(raw_data_path):
            for sub in os.listdir(raw_data_path):  # Iterate through sub-stroke#### directories
                sub_path = os.path.join(raw_data_path, sub, "ses-01")
                if os.path.exists(sub_path):
                    ncct_files.extend(glob.glob(os.path.join(sub_path, "*ncct*.nii*")))  # Finds .nii and .nii.gz

        # Search inside derivatives for mask files
        derivatives_path = os.path.join(batch_path, "derivatives")
        if os.path.exists(derivatives_path):
            for sub in os.listdir(derivatives_path):  # Iterate through sub-stroke#### directories
                sub_path = os.path.join(derivatives_path, sub, "ses-02")
                if os.path.exists(sub_path):
                    mask_files.extend(glob.glob(os.path.join(sub_path, "*msk*.nii*")))  # Finds .nii and .nii.gz

    return ncct_files, mask_files

# Define the root directory where ISLES-2024 dataset is stored
root_directory = "Y:/DATASETS/ISLES-2024"

# Get lists of NCCT and mask file paths
ncct_list, mask_list = find_nii_files(root_directory)

# Print sample results
print(f"Found {len(ncct_list)} NCCT scans")
print(f"Found {len(mask_list)} Mask files")
print("Example NCCT file:", ncct_list[0] if ncct_list else "None")
print("Example Mask file:", mask_list[0] if mask_list else "None")

# # Print all NCCT file paths
# print("\nNCCT Files:")
# for i, file in enumerate(ncct_list):
#     print(f"{i+1}: {file}")

# # Print all Mask file paths
# print("\nMask Files:")
# for i, file in enumerate(mask_list):
#     print(f"{i+1}: {file}")
    
# Split NCCT and Mask lists while maintaining pairing
ncct_train, ncct_test, mask_train, mask_test = train_test_split(
    ncct_list, mask_list, test_size=0.2, random_state=42
)

# Print the sizes of each set
print(f"Total NCCTs: {len(ncct_list)}, Training: {len(ncct_train)}, Testing: {len(ncct_test)}")
print(f"Total Masks: {len(mask_list)}, Training: {len(mask_train)}, Testing: {len(mask_test)}")

