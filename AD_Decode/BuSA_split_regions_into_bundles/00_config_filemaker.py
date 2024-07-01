import os
from DTC.file_manager.computer_nav import write_parameters_to_ini, read_parameters_from_ini


project_headfile_folder = '/Users/jas/bass/gitfolder/DTC_private/Bundle_project_heafile'
project_run_identifier = '202311_10template_test01'

project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')

"""
#previous method used in extracting the template subjects
text_folder = '/Users/jas/jacques/AD_Decode/AD_Decode_bundlesplit/Texts/'
project_run_identifier_txt = os.path.join(text_folder,f'{project_run_identifier}.txt')

with open(project_run_identifier_txt, 'r') as file:
    template_subjects = file.read()

template_subjects = template_subjects.split(',')
template_subjects = [subj.replace(' ','') for subj in template_subjects]
"""

overwrite=True

if not os.path.exists(project_summary_file) or overwrite:

    params = {
        'project' : 'AD_Decode',
        'streamline_type' : 'mrtrix',
        'text_folder' : '/Users/jas/jacques/AD_Decode/AD_Decode_bundlesplit/Texts/',
        'test' : True,
        'ratio' : 100,
        'stepsize' : 2,
        'num_points': 50,
        'distance' : 50,
        'num_bundles' : 6,
        'verbose' : True,
        'setpoints' : True,
        'saveflip' : False,
        'figures_outpath' : '/Users/jas/jacques/Figures_ADDecode',
        'template_subjects' : ["S01912", "S02110", "S02224", "S02227", "S02231", "S02266", "S02289", "S02320", "S02361",
                             "S02363"],
        'added_subjects' : ["S02373", "S02386", "S02390", "S02402", "S02410", "S02421", "S02424", "S02446", "S02451",
                            "S02469", "S02473", "S02485", "S02490", "S02491", "S02506", "S02523", "S02524", "S02535",
                            "S02654", "S02666", "S02670", "S02686", "S02690", "S02695", "S02715", "S02720", "S02737",
                            "S02745", "S02753", "S02765", "S02771", "S02781", "S02802", "S02804", "S02813", "S02812",
                            "S02817","S02840", "S02842", "S02871", "S02877", "S02898", "S02926", "S02938", "S02939",
                            "S02954", "S02967", "S02987", "S03010", "S03017", "S03028", "S03033", "S03034", "S03045",
                            "S03048", "S03069", "S03225","S03265", "S03293", "S03308", "S03321", "S03343", "S03350",
                            "S03378", "S03391", "S03394"],
        'removed_list' : ["S02745","S02230","S02490","S02523",'S02654'],
        'references' : ['fa', 'ln']
    }
    write_parameters_to_ini(project_summary_file, params)


project_run_identifier = '202311_10template_test02_configtest'

project_summary_file = os.path.join(project_headfile_folder,project_run_identifier+'.ini')

if not os.path.exists(project_summary_file) or overwrite:

    params = {
        'project' : 'AD_Decode',
        'streamline_type' : 'mrtrix',
        'text_folder' : '/Users/jas/jacques/AD_Decode/AD_Decode_bundlesplit/Texts/',
        'test' : False,
        'ratio' : 100,
        'stepsize' : 2,
        'num_points': 50,
        'distance' : 50,
        'num_bundles' : 6,
        'verbose' : True,
        'setpoints' : True,
        'saveflip' : False,
        'figures_outpath' : '/Users/jas/jacques/Figures_ADDecode',
        'template_subjects' : ["S01912", "S02110", "S02224", "S02227", "S02231", "S02266", "S02289", "S02320", "S02361",
                             "S02363"],
        'added_subjects' : ["S02373", "S02386", "S02390", "S02402", "S02410", "S02421", "S02424", "S02446", "S02451",
                            "S02469", "S02473", "S02485", "S02490", "S02491", "S02506", "S02523", "S02524", "S02535",
                            "S02654", "S02666", "S02670", "S02686", "S02690", "S02695", "S02715", "S02720", "S02737",
                            "S02745", "S02753", "S02765", "S02771", "S02781", "S02802", "S02804", "S02813", "S02812",
                            "S02817","S02840", "S02842", "S02871", "S02877", "S02898", "S02926", "S02938", "S02939",
                            "S02954", "S02967", "S02987", "S03010", "S03017", "S03028", "S03033", "S03034", "S03045",
                            "S03048", "S03069", "S03225","S03265", "S03293", "S03308", "S03321", "S03343", "S03350",
                            "S03378", "S03391", "S03394"],
        'removed_list' : ["S02745","S02230","S02490","S02523",'S02654'],
        'references': ['fa', 'ln']
    }
    write_parameters_to_ini(project_summary_file, params)


