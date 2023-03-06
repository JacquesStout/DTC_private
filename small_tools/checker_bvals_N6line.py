import os, glob, re
from DTC.diff_handlers.bvec_handler import writebfiles, fix_bvals_bvecs
import numpy as np

subjects = ['N60103', 'N60062', 'N60101', 'N60056', 'N60088', 'N60064', 'N60070', 'N60093', 'N60097', 'N60095', 'N60068', 'N60072', 'N60058', 'N60092', 'N60060', 'N60188', 'N60190', 'N60192', 'N60194', 'N60198', 'N60219', 'N60221', 'N60223', 'N60225', 'N60229', 'N60231']
#subjects = ['N60127','N60129','N60131','N60133','N60137','N60139','N60157','N60159','N60161','N60163','N60167','N60169']
#subjects = ['N59141']
outpath = '/Volumes/Data/Badea/Lab/jacques/'

for subject in subjects:
    folder = f'/Volumes/dusom_civm-atlas/18.abb.11/{subject}_m*'

    direction_folders = glob.glob(folder)
    direction_folders.sort()

    print(direction_folders)

    bvals = []
    bvecs = []

    for direction_folder in direction_folders:
        headfile = glob.glob(os.path.join(direction_folder,'*.headfile'))[0]

        with open(headfile, 'rb') as source:
            i=0
            for line in source:
                #            pattern1 = f'Vector\[{str(i)}\]'
                pattern1 = f'bvalue'
                pattern2 = f'bval_dir'
                rx1 = re.compile(pattern1, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                rx2 = re.compile(pattern2, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for a in rx1.findall(str(line)):
                    if not 'z_Agilent' in str(line):
                        bval = str(line).split('=')[1].split('\\')[0]
                        bvals.append(int(bval))
                        i += 1
                    else:
                        bvals =str(line).split(',')[1].split('\\')[0].split(' ')

                for a in rx2.findall(str(line)):
                    bvec = str(line).split('=')[1].split('\\')[0]
                    bvecs.append(bvec.split('3:1,')[1].split(' '))
                    i += 1
    bvecs = np.array(bvecs)
    #bval_path, bvec_path = writebfiles(bvals, bvecs, outpath, subject, writeformat = "dsi", overwrite=True)
    bval_path, bvec_path = writebfiles(bvals, bvecs, outpath, subject, writeformat="classic", overwrite=True)
    #fbvals, fbvecs = fix_bvals_bvecs(bval_path, bvec_path)
    print(f'Wrote {bval_path} and {bvec_path}')
    #print(bvals)
    #print(bvecs)



