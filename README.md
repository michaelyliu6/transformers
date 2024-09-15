# Connecting to a GPU

1. Go to https://cloud.lambdalabs.com/instances
2. Get an A10 GPU (or whatever is cheapest)
3. Run ssh `ssh -i path/to/your/key.pem username@your-instance-ip-address`

# Two Way Sync set up (for Windows)

Install WinSCP:
WinSCP is a popular SFTP client for Windows that also includes synchronization features.

Download and install WinSCP from the official website: https://winscp.net/

Set up a WinSCP session:
a. Open WinSCP
b. Click "New Session"
c. Fill in the following details:

File protocol: SFTP
Host name: Your Lambda Cloud instance IP
Port number: 22 (default for SSH)
User name: ubuntu (or your Lambda Cloud username)
d. In the "Advanced" settings, go to SSH > Authentication and browse to select your private key file (.pem)
e. Save the session for future use

Configure synchronization:
a. Connect to your server using the session you just created
b. In WinSCP, go to Commands > Keep Remote Directory up to Date
c. In the dialog that appears:

For "Local directory", choose your local workspace folder
For "Remote directory", choose the directory on your Lambda instance where you want to sync files
d. Click "OK" to start the initial synchronization

Enable real-time synchronization:
a. In WinSCP, go to Options > Preferences
b. Navigate to Endurance > Background
c. Check the box for "Keep remote directory up to date"
d. Set "Reconnect session when lost" to a value like 5 seconds
e. Click "OK" to save these settings

# Install Dependencies

pip install transformers
pip install tiktoken
pip install --upgrade networkx
pip install --upgrade torch
