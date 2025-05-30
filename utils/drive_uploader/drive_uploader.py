import os
import pickle
from urllib.request import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

class GoogleDriveClient:
    """A class to handle Google Drive operations."""
    
    # If modifying these SCOPES, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive']

    def __init__(self, credentials_file='credentials.json', token_file='token.pickle', default_folder_id='1KrM53zogG5-62_5qF2zRxcSUZKuds1_L'):
        """Initialize the Google Drive client.
        
        Args:
            credentials_file: Path to the credentials.json file.
            token_file: Path to the token.pickle file.
            default_folder_id: Default folder ID to use for uploads.
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.default_folder_id = default_folder_id
        self.service = self._get_drive_service()
    
    def _get_drive_service(self):
        """Get an authorized Google Drive service instance."""
        creds = None

        # The file token.pickle stores the user's access and refresh tokens
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)

        return build('drive', 'v3', credentials=creds)
    
    def upload_file(self, file_path, folder_id=None):
        """Upload a file to Google Drive.
        
        Args:
            file_path: Path to the file to upload.
            folder_id: ID of the folder to upload to. If None, uses default_folder_id.
        
        Returns:
            File ID of the uploaded file.
        """
        folder_id = folder_id or self.default_folder_id
        file_name = os.path.basename(file_path)
        file_metadata = {'name': file_name}
        
        if folder_id:
            file_metadata['parents'] = [folder_id]
            
        media = MediaFileUpload(file_path)
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        print(f"File '{file_name}' uploaded with ID: {file['id']}")
        return file['id']
    
    def create_folder(self, folder_name, parent_folder_id=None):
        """Create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder to create.
            parent_folder_id: ID of the parent folder. If None, uses default_folder_id.
            
        Returns:
            Folder ID of the created folder.
        """
        parent_folder_id = parent_folder_id or self.default_folder_id
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_folder_id:
            folder_metadata['parents'] = [parent_folder_id]
            
        folder = self.service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        
        print(f"Folder '{folder_name}' created with ID: {folder['id']}")
        return folder['id']
    
    def upload_folder(self, folder_path, parent_folder_id=None):
        """Upload an entire folder to Google Drive recursively.
        
        Args:
            folder_path: Path to the folder to upload.
            parent_folder_id: ID of the parent folder. If None, uses default_folder_id.
            
        Returns:
            Folder ID of the uploaded folder.
        """
        parent_folder_id = parent_folder_id or self.default_folder_id
        folder_name = os.path.basename(folder_path)
        
        # Create a new folder in Google Drive
        drive_folder_id = self.create_folder(folder_name, parent_folder_id)
        
        # Upload all files and folders in the directory
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            if os.path.isfile(item_path):
                # Upload file
                self.upload_file(item_path, drive_folder_id)
            elif os.path.isdir(item_path):
                # Recursively upload subfolder
                self.upload_folder(item_path, drive_folder_id)
        
        return drive_folder_id
    
    def replace_file(self, file_path):
        """Replace local file with a txt containing the file ID on Google Drive.

        Args:
            file_path: Path to the local file to replace.
        """    
        # Upload the new file to Google Drive
        file_id = self.upload_file(file_path)

        # get path of the file and its name
        path_only = os.path.dirname(file_path)
        file_name = os.path.basename(file_path) + '.txt'

        # replace the file in the local folder
        os.remove(file_path)
        with open(os.path.join(path_only, file_name), 'w') as f:
            f.write(file_id)

        print(f"File '{file_name}' replaced with file ID: {file_id}")
    
    def download_file(self, path):
        """Download a file from Google Drive. The file ID is stored in a text file.
        
        Args:
            path: Path to a text file containing the file ID.
        """
        
        # Read the file ID from the text file
        with open(path, 'r') as f:
            file_id = f.read().strip()
        
        # Create a request to download the file
        request = self.service.files().get_media(fileId=file_id)
        
        output_file = path.replace('.txt', '')
        with open(output_file, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")
        
        print(f"File downloaded as '{output_file}'.")