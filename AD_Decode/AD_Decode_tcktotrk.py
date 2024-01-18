import sys, re, subprocess
from DTC.tract_manager.tract_save import convert_tck_to_trk

def get_num_streamlines(tracks_path):
    cmd = f'tckinfo {tracks_path} -count'
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        # Split the output into lines and extract the last line
        output_lines = result.stdout.split('\n')
        last_line = output_lines[-2] if output_lines else ''
        numbers = re.findall(r'\d+', last_line)
        num_streamlines = max(map(int, numbers))
        return num_streamlines
    else:
        # Handle the case where the command failed
        print(f"Error: {result.stderr}")
        return None

tck_file = sys.argv[1]
trk_file = sys.argv[2]
reference_file = sys.argv[3]
num_min = sys.argv[4]

num_streamlines = get_num_streamlines(tck_file)
print(num_streamlines)

if int(num_streamlines)<int(num_min):
    raise Exception('Tck file did not have minimum number of streamlines')
else:
    convert_tck_to_trk(tck_file, trk_file,reference_file)

