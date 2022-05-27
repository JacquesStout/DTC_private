import sys
import os
from file_tools import getfromfile

def main():
    print(os.name)

if __name__ == '__main__':
    try:
        if sys.argv[1] == 'deploy':
            import paramiko

            # Connect to remote host
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            username, passwd = getfromfile('/Users/jas/samos_connect.rtf')
            client.connect('remote_hostname_or_IP', username='alex', password='4.alex')

            # Setup sftp connection and transmit this script
            sftp = client.open_sftp()
            __file__ = '/Users/jas/bass/gitfolder/wuconnectomes/streamline_bundle_stats.py'
            sftp.put(__file__, '/tmp/myscript.py')
            sftp.close()

            # Run the transmitted script remotely without args and show its output.
            # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
            stdout = client.exec_command('python3 /tmp/myscript.py')[1]
            for line in stdout:
                # Process each line in the remote output
                print(line)

            client.close()
            sys.exit(0)
    except IndexError:
        pass