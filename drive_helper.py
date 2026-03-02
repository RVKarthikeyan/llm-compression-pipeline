import os
import argparse
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
# This scope allows the app to view and manage files it creates.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    """Handles Google Drive OAuth2.0 Authentication."""
    creds = None
    # token.json stores the user's access and refresh tokens. It is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("Error: 'credentials.json' not found. Please download it from Google Cloud Console.")
                exit(1)
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    return creds

def upload_file(file_path, drive_name=None, folder_id=None):
    """Uploads a file to Google Drive."""
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    creds = authenticate()

    try:
        # Build the Drive API service
        service = build('drive', 'v3', credentials=creds)

        # Use provided name or default to the file's base name
        file_name = drive_name if drive_name else os.path.basename(file_path)

        file_metadata = {'name': file_name}
        
        # If a folder ID is provided, append it to the metadata parents list
        if folder_id:
            file_metadata['parents'] = [folder_id]

        # Read the file and let googleapiclient guess the mime type
        media = MediaFileUpload(file_path, resumable=True)

        print(f"Uploading '{file_name}' to Google Drive...")
        
        # Execute the upload request
        file = service.files().create(
            body=file_metadata, 
            media_body=media, 
            fields='id'
        ).execute()

        print(f"Success! File ID: {file.get('id')}")

    except HttpError as error:
        print(f"An error occurred: {error}")

def main():
    parser = argparse.ArgumentParser(description="Upload files to Google Drive via CLI.")
    
    # Positional argument (Required)
    parser.add_argument("filepath", help="The local path to the file you want to upload.")
    
    # Optional arguments
    parser.add_argument("-n", "--name", help="Optional: A new name for the file on Google Drive.", default=None)
    parser.add_argument("-f", "--folder", help="Optional: The ID of the Google Drive folder to upload to.", default=None)

    args = parser.parse_args()

    upload_file(args.filepath, args.name, args.folder)

if __name__ == '__main__':
    main()
