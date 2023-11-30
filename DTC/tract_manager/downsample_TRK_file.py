import sys, os
from DTC.file_manager.computer_nav import get_mainpaths
from DTC.file_manager.file_tools import mkcdir, check_files, getfromfile
from DTC.tract_manager.tract_handler import reducetractnumber

fileold = sys.argv[1]
filenew = sys.argv[2]
ratio = sys.argv[3]
method = sys.argv[4]

remote=True
if remote:
    username, passwd = getfromfile(os.path.join(os.environ['HOME'],'remote_connect.rtf'))
else:
    username = None
    passwd = None
inpath, _, _, sftp = get_mainpaths(remote,project = 'AD_Decode', username=username,password=passwd)

reducetractnumber(fileold, filenew, getdata=False, ratio=ratio, method=method, return_affine=False, verbose=False,
                  sftp=sftp)