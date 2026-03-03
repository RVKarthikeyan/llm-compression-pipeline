import os
import argparse
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
# Added drive.readonly to allow listing and downloading any file.
SCOPES = ['https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive.readonly']

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

def list_files(page_size=10, query=None):
    """Lists the files in Google Drive."""
    creds = authenticate()
    try:
        service = build('drive', 'v3', credentials=creds)
        
        print("Fetching files...")
        results = service.files().list(
            pageSize=page_size, 
            fields="nextPageToken, files(id, name, mimeType)",
            q=query
        ).execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print(f"- {item['name']} (ID: {item['id']}, Type: {item['mimeType']})")
    except HttpError as error:
        print(f"An error occurred: {error}")

def download_file(file_id, output_path):
    """Downloads a file from Google Drive."""
    creds = authenticate()
    try:
        service = build('drive', 'v3', credentials=creds)

        # Get file metadata to get the name if output_path is a directory
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata.get('name')

        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, file_name)

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        print(f"Downloading '{file_name}'...")
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {int(status.progress() * 100)}%.")
        
        with open(output_path, 'wb') as f:
            f.write(fh.getbuffer())
        
        print(f"Success! File saved to '{output_path}'")

    except HttpError as error:
        print(f"An error occurred: {error}")

def main():
    parser = argparse.ArgumentParser(description="Google Drive CLI Helper.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a file to Google Drive.")
    upload_parser.add_argument("filepath", help="The local path to the file you want to upload.")
    upload_parser.add_argument("-n", "--name", help="Optional: A new name for the file on Google Drive.", default=None)
    upload_parser.add_argument("-f", "--folder", help="Optional: The ID of the Google Drive folder to upload to.", default=None)

    # List command
    list_parser = subparsers.add_parser("list", help="List files on Google Drive.")
    list_parser.add_argument("-p", "--pagesize", type=int, default=10, help="Number of files to list.")
    list_parser.add_argument("-q", "--query", help="Search query (e.g., \"name contains 'model'\").")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a file from Google Drive.")
    download_parser.add_argument("file_id", help="The ID of the file to download.")
    download_parser.add_argument("-o", "--output", default=".", help="The local path or directory to save the file.")

    args = parser.parse_args()

    if args.command == "upload":
        upload_file(args.filepath, args.name, args.folder)
    elif args.command == "list":
        list_files(args.pagesize, args.query)
    elif args.command == "download":
        download_file(args.file_id, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
