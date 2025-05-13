import subprocess
import os
import shutil

RCLONE_REMOTE_NAME = "onedrive"  # remote name as configured in rclone.config
ONEDRIVE_BASE_DESTINATION_PATH = "AI-project/"

def move_folder_to_onedrive(local_folder_path, onedrive_subfolder_name=None, delete_local_folder=True, verbose=False):
    """
    Move local folder to OneDrive using rclone.

    Args:
        local_folder_path (str): Local path of folder to move.
        onedrive_subfolder_name (str, optional): Name of OneDrive subfolder. If not specified local folder name is used.
    Returns:
        bool: True if operation was successful, False otherwise.
    """

    if not shutil.disk_usage(os.path.dirname(os.path.abspath(local_folder_path))):
        print(f"ERROR: The local path '{local_folder_path}' isn't valid.")
        return False

    if not os.path.isdir(local_folder_path):
        print(f"ERROR: '{local_folder_path}' is not a valid folder.")
        return False

    folder_name = os.path.basename(local_folder_path.rstrip('/\\'))
    if onedrive_subfolder_name:
        destination_folder_name = onedrive_subfolder_name
    else:
        destination_folder_name = folder_name

    # Create the destination folder on OneDrive if it doesn't exist
    # Makes sure there are no double slashes if ONEDRIVE_BASE_DESTINATION_PATH is empty
    if ONEDRIVE_BASE_DESTINATION_PATH and not ONEDRIVE_BASE_DESTINATION_PATH.endswith('/'):
        base_dest = ONEDRIVE_BASE_DESTINATION_PATH + '/'
    else:
        base_dest = ONEDRIVE_BASE_DESTINATION_PATH


    # Define the OneDrive parent path where we'll check for existing folders
    onedrive_parent_path_str = f"{RCLONE_REMOTE_NAME}:{base_dest}"
    base_folder_name_on_onedrive = destination_folder_name


    # --- Check if the destination folder already exists --- #
    print(f"Checking if the destination folder '{base_folder_name_on_onedrive}' already exists on OneDrive...")
    try:
        # Command to list only directories in the base destination on OneDrive
        # 'lsf --dirs-only' returns directory names, one per line, ending with '/'
        list_command = [
            "rclone",
            "lsf",
            onedrive_parent_path_str,
            "--dirs-only"
        ]
        list_process = subprocess.Popen(list_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_list, stderr_list = list_process.communicate()

        if list_process.returncode != 0:
            print(f"ERROR listing folders on OneDrive (stdout): {stdout_list}")
            print("Trying to move with the original name, but there may be conflicts.")
            existing_folders_on_onedrive = []  # Fallback to empty list
        else:
            # Clean the output: remove '/' at the end and split by lines
            existing_folders_on_onedrive = [name.rstrip('/') for name in stdout_list.strip().split('\n') if name]

    except FileNotFoundError:
        print("ERROR: rclone command not found. Make sure rclone is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while listing folders: {e}")
        return False

    # Find a unique folder name if the base name already exists
    final_folder_name_on_onedrive = base_folder_name_on_onedrive
    suffix_counter = 1
    while final_folder_name_on_onedrive in existing_folders_on_onedrive:
        final_folder_name_on_onedrive = f"{base_folder_name_on_onedrive}_{suffix_counter}"
        suffix_counter += 1
        if suffix_counter > 100:
            print("ERROR: Too many folders with the same base name. Please check your OneDrive.")
            return False

    if final_folder_name_on_onedrive != base_folder_name_on_onedrive:
        print(f"Conflict detected: '{base_folder_name_on_onedrive}' already exists. Using '{final_folder_name_on_onedrive}' instead.")
    # --- END OF CHECKING FOR EXISTING FOLDERS --- #


    # Use the final folder name we determined
    onedrive_destination = f"{RCLONE_REMOTE_NAME}:{base_dest}{final_folder_name_on_onedrive}"

    # rclone command to execute
    command = [
        "rclone",
        "move",  # can use "copy" if you want to copy instead of move.
        local_folder_path,
        onedrive_destination,
        "--create-empty-src-dirs",  # creates the directory structure even if some folders are empty
        "-P" # shows progress
    ]

    print(f"Moving '{local_folder_path}' to OneDrive: '{onedrive_destination}'...")
    print(f"rclone command: {' '.join(command)}")

    try:
        # Run rclone command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"SUCCESS: Folder '{folder_name}' moved successfully to OneDrive.")
            
            if delete_local_folder:
                # Remove the local folder after successful move
                shutil.rmtree(local_folder_path)

            if stdout and verbose:
                print("Output rclone:\n", stdout)
            return True
        else:
            print(f"ERROR: rclone returned an error code {process.returncode}.")
            if stdout:
                print("rclone output (stdout):\n", stdout)
            if stderr:
                print("rclone errors (stderr):\n", stderr)
            return False
    except FileNotFoundError:
        print("ERROR: Command rclone not found. Make sure rclone is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return False

def download_folder_from_onedrive(onedrive_folder_path, local_destination_path):
    """
    Download a folder from OneDrive to a local destination using rclone.

    Args:
        onedrive_folder_path (str): Path of the folder on OneDrive.
        local_destination_path (str): Local path where the folder will be downloaded.
    Returns:
        bool: True if operation was successful, False otherwise.
    """
    command = [
        "rclone",
        "copy",
        f"{RCLONE_REMOTE_NAME}:{ONEDRIVE_BASE_DESTINATION_PATH}{onedrive_folder_path}",
        local_destination_path,
        "-P"  # shows progress
    ]

    print(f"Downloading '{onedrive_folder_path}' from OneDrive to '{local_destination_path}'...")
    print(f"rclone command: {' '.join(command)}")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("SUCCESS: Folder downloaded successfully.")
            return True
        else:
            print(f"ERROR: rclone returned an error code {process.returncode}.")
            if stdout:
                print("rclone output (stdout):\n", stdout)
            if stderr:
                print("rclone errors (stderr):\n", stderr)
            return False
    except FileNotFoundError:
        print("ERROR: Command rclone not found. Make sure rclone is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return False

def update_folder_on_onedrive(onedrive_folder_path, local_folder_path):
    """
    Update a folder on OneDrive with the contents of a local folder using rclone.

    Args:
        onedrive_folder_path (str): Path of the folder on OneDrive.
        local_folder_path (str): Local path of the folder to upload.
    Returns:
        bool: True if operation was successful, False otherwise.
    """
    command = [
        "rclone",
        "sync",
        local_folder_path,
        f"{RCLONE_REMOTE_NAME}:{ONEDRIVE_BASE_DESTINATION_PATH}{onedrive_folder_path}",
        "-P"  # shows progress
    ]

    print(f"Updating '{onedrive_folder_path}' on OneDrive with '{local_folder_path}'...")
    print(f"rclone command: {' '.join(command)}")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("SUCCESS: Folder updated successfully.")

            # delete the local folder after successful update
            if os.path.isdir(local_folder_path):
                shutil.rmtree(local_folder_path)
                print(f"Local folder '{local_folder_path}' has been removed after successful update.")
            else:
                print(f"WARNING: Local folder '{local_folder_path}' does not exist after update.")
            return True
        else:
            print(f"ERROR: rclone returned an error code {process.returncode}.")
            if stdout:
                print("rclone output (stdout):\n", stdout)
            if stderr:
                print("rclone errors (stderr):\n", stderr)
            return False
    except FileNotFoundError:
        print("ERROR: Command rclone not found. Make sure rclone is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        return False


# --- EXAMPLE USAGE ---
# local_folder_to_move = "./my_folder"
# success = move_folder_to_onedrive(local_folder_to_move)
# if success:
#     print("Completed.")
#     # Verify the local folder was removed
#     if not os.path.exists(local_folder_to_move):
#         print(f"Local folder '{local_folder_to_move}' has been successfully removed.")
#     else:
#         print(f"WARNING: Local folder '{local_folder_to_move}' still exists.")
# else:
#     print("Failed to move the folder. Check the error messages above.")
#
# success = download_folder_from_onedrive("AI-project/my_folder", "./downloaded_folder")
# if success:
#     print("Downloaded successfully.")
# else:
#     print("Failed to download the folder. Check the error messages above.")
#
# success = update_folder_on_onedrive("AI-project/my_folder", "./my_folder")
# if success:
#     print("Updated successfully.")
# else:
#     print("Failed to update the folder. Check the error messages above.")
# --- END OF EXAMPLE USAGE ---