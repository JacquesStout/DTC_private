import getpass
import os
from pathlib import Path
import shutil
from DTC.file_manager.file_tools import buildlink, mkcdir, getfromfile
import glob
import numpy as np
import warnings
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import create_backport_labels, create_MDT_labels
import os
from DTC.file_manager.computer_nav import get_mainpaths, checkfile_exists_remote, getremotehome
from DTC.nifti_handlers.atlas_handlers.convert_atlas_mask import convert_labelmask, atlas_converter
from DTC.file_manager.argument_tools import parse_arguments
from DTC.nifti_handlers.atlas_handlers.create_backported_labels import get_info_SAMBA_headfile

#project = "APOE"
#project = "AMD"
project = 'AD_Decode'
#project = 'ADRC'
#project = 'Chavez'
verbose = True
mainpath = getremotehome('Lab')
SAMBA_headfile_dir = os.path.join(mainpath, "samba_startup_cache")
file_ids = ["subjspace_coreg","coreg", "subjspace_fa", "subjspace_b0", "bval", "bvec", "subjspace_mask", "reference", "subjspace_dwi", "relative_orientation"]
file_ids = ["subjspace_coreg","bval", "bvec", "subjspace_mask", "subjspace_dwi", "relative_orientation"]
file_ids = []
#file_ids = ["subjspace_coreg"]
#file_ids = ["subjspace_mask"]
#file_ids = ['subjspace_dwi']
#file_ids = ['subjspace_mask']
#if project == 'AMD':
#    file_ids = ["relative_orientation"]

mymax=-1
identifier_SAMBA_folder = ''

if project == 'Chavez':
    SAMBA_mainpath = os.path.join(mainpath, "mouse")

    #SAMBA_projectname = "VBM_20APOE01_chass_symmetric3_allAPOE"
    #SAMBA_projectname = "VBM_21Chavez01_chass_symmetric3_all"
    SAMBA_projectname = "VBM_21Chavez02_chass_symmetric3_all"

    SAMBA_headfile = os.path.join(SAMBA_headfile_dir, "jas297_SAMBA_Chavez_set.headfile")

    gunniespath = "~/gunnies/"
    recenter = 0
    # SAMBA_prep_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname+"-inputs")
    #SAMBA_prep_folder = os.path.join(mainpath, "APOE_symlink_pool")
    #SAMBA_prep_folder = os.path.join(mainpath, '19abb14')
    SAMBA_prep_folder = os.path.join(mainpath, "mouse","Chavez_symlink_pool_allfiles")

    atlas_labels = os.path.join(mainpath,"atlases","chass_symmetric3","chass_symmetric3_labels.nii.gz")
    atlas_name = 'chass_symmetric3'
    atlas_legends = os.path.join(mainpath,'atlases/CHASSSYMM3AtlasLegends.xlsx')
    DTC_DWI_folder = "DWI"
    DTC_labels_folder = "DWI"

    SAMBA_label_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-results", "connectomics")
    SAMBA_work_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-work")
    orient_string = os.path.join(SAMBA_prep_folder, "relative_orientation.txt")
    superpose = False
    copytype = "truecopy"
    overwrite = False
    preppath = None
    subjects = ['apoe_3_6_CtrlA_male', 'MHI_335_CtrA_male', 'apoe_4_4_CtrlB_female','MHI_326_C_male','MHI_334_CtrC_female',
                'MHI_335_CtrB_female','MHI_326_D_female','apoe_4_2_A_male','apoe_3_4_CtrA_female','apoe_4_7_B_male',
                'MHI_326_CtrB_male','MHI_334_D_female','apoe_4_7_A_female','apoe_4_2_B_female','apoe_3_6_B_male',
                'MHI_334_CtrA_male','apoe_4_5_CtrA_female','apoe_4_4_CtrlD_male','apoe_4_4_B_female','MHI_326_CtrA_female',
                'apoe_3_4_A_female']

    #subjects_folders = glob.glob(os.path.join(SAMBA_work_folder, '*affine.mat/'))

    """
    subjects_all = glob.glob(os.path.join(SAMBA_work_folder, 'preprocess','*_dwi_masked.nii.gz'))
    subjects = []
    for subject in subjects_all:
        subject_name = os.path.basename(subject)
        subjects.append(subject_name[:6])
    """

    #removed_list = ['C_20220124_004'] #004 had a terrible orientation and would have to be individually treated in order to be on par with the others
    removed_list = []
    for remove in removed_list:
        if remove in subjects:
            subjects.remove(remove)


elif project == "AD_Decode":

    SAMBA_mainpath = os.path.join(mainpath, "mouse")
    SAMBA_projectname = "VBM_21ADDecode03_IITmean_RPI_fullrun"
    SAMBA_headfile = os.path.join(SAMBA_headfile_dir, "jas297_SAMBA_ADDecode.headfile")
    gunniespath = "~/gunnies/"
    recenter = 0
    #SAMBA_prep_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname+"-inputs")
    #SAMBA_prep_folder = os.path.join(mainpath, "human","ADDeccode_symlink_pool_allfiles")
    #SAMBA_prep_folder = os.path.join(mainpath, "mouse", "ADDeccode_symlink_pool2")
    SAMBA_prep_folder = os.path.join(mainpath, "human", "ADDeccode_symlink_pool_allfiles")
    atlas_labels = os.path.join(mainpath, "atlas","IITmean_RPI","IITmean_RPI_labels.nii.gz")
    atlas_legends = os.path.join(mainpath, "atlases/IITmean_RPI/IITmean_RPI_index.xlsx")
    DTC_DWI_folder = os.path.join(mainpath, "..","ADdecode.01","Analysis","DWI")
    DTC_labels_folder = os.path.join(mainpath, "..","ADdecode.01","Analysis","DWI")
    DTC_transforms = os.path.join(mainpath, "..","ADdecode.01","Analysis","Transforms")
    atlas_name = 'IITmean_RPI'
    DTC_DWI_folder = "DWI"
    DTC_labels_folder = "DWI"

    SAMBA_work_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-work")
    SAMBA_label_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname+"-results", "connectomics")
    orient_string = os.path.join(SAMBA_prep_folder,"relative_orientation.txt")
    superpose=True
    copytype="truecopy"
    overwrite=False
    preppath = None
    subjects = ["S02654", "S02666",  "S02670",  "S02686", "S02690", "S02695",  "S02715", "S02720", "S02737", "S02753", "S02765", "S02771", "S02781", "S02802",
                "S02804", "S02813", "S02817", "S02840", "S02877", "S02898", "S02926", "S02938", "S02939", "S02954", "S02967",
                "S02987", "S03010", "S03017", "S03033", "S03034", "S03045", "S03048"]
    subjects = ["S02802","S01912", "S02110", "S02224", "S02227", "S02230", "S02231", "S02266", "S02289", "S02320", "S02361", "S02363", "S02373", "S02386", "S02390", "S024S02", "S02410", "S02421", "S02424", "S02446", "S02451", "S02469", "S02473", "S02485", "S02490", "S02491", "S02506","S02524","S02535","S02690","S02715","S02771","S02804","S02817", "S02840","S02877","S02898","S02926","S02938","S02939","S02954", "S03017", "S03028", "S03048","S02524","S02535","S02690","S02715","S02771","S02804","S02812","S02817", "S02840","S02871","S02877","S02898","S02926","S02938","S02939","S02954", "S03017", "S03028", "S03048", "S03069","S02817"]

    subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320', 'S02361', 'S02363',
                'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451', 'S02469', 'S02473',
                'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654', 'S02666', 'S02670', 'S02686',
                'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753', 'S02765', 'S02771', 'S02781', 'S02802',
                'S02804', 'S02813', 'S02812', 'S02817', 'S02840', 'S02842', 'S02871', 'S02877', 'S02898', 'S02926', 'S02938',
                'S02939', 'S02954', 'S02967', 'S02987', 'S03010', 'S03017', 'S03028', 'S03033', 'S03034', 'S03045', 'S03048',
                'S03069', 'S03225', 'S03265', 'S03293', 'S03308', 'S03321', 'S03343', 'S03350', 'S03378', 'S03391', 'S03394']

    subjects = ["S03343", "S03350", "S03378", "S03391", "S03394"]

    #subjects = ["S02695", "S02686", "S02670", "S02666", "S02654","S03010"]
    subjects = ["S01912"]
    subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320', 'S02361',
                'S02363',
                'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451', 'S02469',
                'S02473',
                'S02485', 'S02491', 'S02490', 'S02506']
    subjects = ['S03847', 'S03866', 'S03867', 'S03889', 'S03890', 'S03896']

    subjects_all = glob.glob(os.path.join(SAMBA_work_folder, 'preprocess','*_dwi_masked.nii.gz'))
    subjects = []
    for subject in subjects_all:
        subject_name = os.path.basename(subject)
        subjects.append(subject_name[:6])

    subjects = ['T01257', 'T01277' ,'T01402', 'T04086', 'T04129', 'T04300' ,'T04472', 'T01501', 'T01516', 'T01541', 'T04602']
    subjects = ['T04129']
    subjects = ['S00775', 'S04491', 'S04493', 'S01412', 'S04526', 'S01470', 'S01619', 'S01620',
                'S01621', 'S04696', 'S04738','S02842']
    subjects = ['S01912', 'S02110', 'S02224', 'S02227', 'S02230', 'S02231', 'S02266', 'S02289', 'S02320', 'S02361', 'S02363',
                'S02373', 'S02386', 'S02390', 'S02402', 'S02410', 'S02421', 'S02424', 'S02446', 'S02451', 'S02469', 'S02473',
                'S02485', 'S02491', 'S02490', 'S02506', 'S02523', 'S02524', 'S02535', 'S02654', 'S02666', 'S02670', 'S02686',
                'S02690', 'S02695', 'S02715', 'S02720', 'S02737', 'S02745', 'S02753', 'S02765', 'S02771', 'S02781', 'S02802',
                'S02804', 'S02813', 'S02812', 'S02817', 'S02840', 'S02842', 'S02871', 'S02877', 'S02898', 'S02926', 'S02938',
                'S02939', 'S02954', 'S02967', 'S02987', 'S03010', 'S03017', 'S03028', 'S03033', 'S03034', 'S03045', 'S03048',
                'S03069', 'S03225', 'S03265', 'S03293', 'S03308', 'S03321', 'S03343', 'S03350', 'S03378', 'S03391', 'S03394',
                'S03847', 'S03866', 'S03867', 'S03889', 'S03890', 'S03896','S00775', 'S04491', 'S04493', 'S01412', 'S04526',
                'S01470', 'S01619', 'S01620', 'S01621', 'S04696', 'S04738']
    identifier_SAMBA_folder = 'faMDT_NoName'

    removed_list = ['S02230', 'S02490', 'S02745']
    for remove in removed_list:
        if remove in subjects:
            subjects.remove(remove)

    #subjects = ['S03847', 'S03866', 'S03867', 'S03889', 'S03890', 'S03896']


    #oldsubjects = ["01912", "02110", "02224", "02227", "02230", "02231", "02266", "02289", "02320", "02361",
    # #           "02363", "02373", "02386", "02390", "02402", "02410", "02421", "02424", "02446", "02451",
    #            "02469", "02473", "02485", "02490", "02491", "02506"]
    #subjects = []
    #for subject in oldsubjects:
    #    subjects.append('S'+subject)
    # 02842, 03028 has apparently a 92 stack ? to investigate
    #"S02812, , "S02871""  "S03069" has a problem? not prepped

elif project == "APOE":

    SAMBA_mainpath = os.path.join(mainpath, "mouse")
    SAMBA_projectname = "VBM_20APOE01_chass_symmetric3_allAPOE"
    SAMBA_headfile = os.path.join(SAMBA_headfile_dir, "jas297_SAMBA_APOE.headfile")
    gunniespath = "~/gunnies/"
    recenter = 0
    # SAMBA_prep_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname+"-inputs")
    #SAMBA_prep_folder = os.path.join(mainpath, "APOE_symlink_pool")
    #SAMBA_prep_folder = os.path.join(mainpath, '19abb14')


    SAMBA_prep_folder = os.path.join(mainpath, "mouse","APOE_symlink_pool_allfiles")
    #SAMBA_prep_folder = os.path.join('/Volumes/Data/Badea/.snapshot/weekly.2023-02-05_0015/Lab/mouse',"APOE_symlink_pool_allfiles")

    atlas_labels = os.path.join(mainpath,"atlases","chass_symmetric3","chass_symmetric3_labels.nii.gz")
    atlas_name = 'chass_symmetric3'
    atlas_legends = os.path.join(mainpath,'atlases/CHASSSYMM3AtlasLegends.xlsx')

    DTC_DWI_folder = os.path.join(mainpath,"mouse","APOE_series","DWI")
    DTC_labels_folder = os.path.join(mainpath,"mouse","APOE_series","DWI")
    DTC_DWI_folder = "DWI_allsubj/"
    DTC_labels_folder = "DWI_allsubj/"
    SAMBA_label_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-results", "connectomics")
    SAMBA_work_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-work")
    orient_string = os.path.join(SAMBA_prep_folder, "relative_orientation.txt")
    superpose = False
    copytype = "truecopy"
    overwrite = False
    preppath = None
    """
    subjects = ['N57437', 'N57442', 'N57446', 'N57447', 'N57449', 'N57451', 'N57496', 'N57498', 'N57500', 'N57502', 'N57504', 'N57513', 'N57515', 'N57518', 'N57520', 'N57522', 'N57546', 'N57548', 'N57550', 'N57552', 'N57554', 'N57559', 'N57580', 'N57582', 'N57584', 'N57587', 'N57590', 'N57692', 'N57694', 'N57700', 'N57702', 'N57709', 'N58302', ' N58303', ' N58305', ' N58309', ' N58310', ' N58344', 'N58346', 'N58350', 'N58355', 'N58359', 'N58361', 'N58394', 'N58396', 'N58398', 'N58400', 'N58402', 'N58404', 'N58406', 'N58408', 'N58477', 'N58500', 'N58510', 'N58512', 'N58514', 'N58516', 'N58604', 'N58606', 'N58608', 'N58610', 'N58611', 'N58613', 'N58706', 'N58708', 'N58712', ' N58214', 'N58215', 'N58216', 'N58217', 'N58218', 'N58219', 'N58221', 'N58222', 'N58223', 'N58224', 'N58225', 'N58226', 'N58228', 'N58229', 'N58230', 'N58231', 'N58232', 'N58633', 'N58634', 'N58635', 'N58636', 'N58649', 'N58650', 'N58651', 'N58653', 'N58654', ' N58398', ' N58714', ' N58740', ' N58477', ' N58734', ' N58792', ' N58302', ' N58784', ' N58706', ' N58361', ' N58355', ' N58712', ' N58790', ' N58606', ' N58350', ' N58608', ' N58779', ' N58500', ' N58604', ' N58749', ' N58510', ' N58394', ' N58346', ' N58788', ' N58514', ' N58794', ' N58733', ' N58655', ' N58735', ' N58400', ' N58708', ' N58780', ' N58512', ' N58747', ' N58404', ' N58751', ' N58611', ' N58745', ' N58406', ' N58359', ' N58742', ' N58396', ' N58613', ' N58732', ' N58516', ' N58402', ' N58935', ' N59003', ' N58819', ' N58909', ' N58919', ' N58889', ' N59010', ' N58859', ' N58917', ' N58815', ' N58997', ' N58999', ' N58881', ' N58853', ' N58995', ' N58877', ' N58883', ' N58885', ' N58906', ' N58821', ' N58855', ' N58861', ' N58857', ' N58851', ' N58887', ' N58879', ' N58829', ' N58913', ' N58831', ' N58941', ' N58813', ' N58952', ' N59022', ' N59026', ' N59033', ' N59035', ' N59039', ' N59041', ' N59065', ' N59066', ' N59072', ' N59076', ' N59078', ' N59080', ' N59097', ' N59099', ' N59109', ' N59116', ' N59118', ' N59120']
    """
    #subjects_folders = glob.glob(os.path.join(SAMBA_work_folder, '*affine.mat/'))
    subjects_all = glob.glob(os.path.join(SAMBA_work_folder, 'preprocess','*_dwi_masked.nii.gz'))
    subjects = []
    for subject in subjects_all:
        subject_name = os.path.basename(subject)
        subjects.append(subject_name[:6])
    """
    subjects = ['N58408', 'N59072', 'N58610', 'N58398', 'N58935', 'N58714', 'N58740', 'N58477', 'N59003', 'N58734', 'N58309', 'N58792', 'N58819', 'N58302', 'N59078', 'N59116', 'N58909', 'N58612', 'N59136', 'N59140', 'N58784', 'N58919', 'N58706', 'N58889', 'N58361', 'N58355', 'N59066', 'N58712', 'N58790', 'N59010', 'N58859', 'N58946', 'N58917', 'N58606', 'N58815', 'N59118', 'N58997', 'N58350', 'N59022', 'N58999', 'N59141', 'N58881', 'N59026', 'N58608', 'N58853', 'N58779', 'N58995', 'N58500', 'N58604', 'N58749', 'N58877', 'N58883', 'N58915', 'N59109', 'N59120', 'N58510', 'N58885', 'N58906', 'N59065', 'N58394', 'N58821', 'N58855', 'N58346', 'N58861', 'N59005', 'N58344', 'N58954', 'N59099', 'N58857', 'N58788', 'N58305', 'N58514', 'N58851', 'N59076', 'N59097', 'N58794', 'N58733', 'N58655', 'N58887', 'N58735', 'N58310', 'N59035', 'N58879', 'N58400', 'N59041', 'N58952', 'N58708', 'N58780', 'N58512', 'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58829', 'N58913', 'N58745', 'N58831', 'N58406', 'N58359', 'N58742', 'N58396', 'N58948', 'N58941', 'N59033', 'N58732', 'N58516', 'N59080', 'N58813', 'N59039', 'N58402']
    #subjects = ['N58784']
    #subjects = ['N59066']
    #subjects = ['N58612','N59136','N59140','N58946','N59141','N58915','N59005','N58954','N58948']
    subjects = ['N60188', 'N60190', 'N60192', 'N60194', 'N60198', 'N60219', 'N60221', 'N60223', 'N60225', 'N60229', 'N60231']
    subjects = ['N57442', 'N57504', 'N58400', 'N58611', 'N58613', 'N58859', 'N60101', 'N60056', 'N60064', 'N60092', 'N60088', 'N60093', 'N60097', 'N60095', 'N60068', 'N60072', 'N60058', 'N60103', 'N60060', 'N60190', 'N60225', 'N60198', 'N58612', 'N60062', 'N60070', 'N60221', 'N60223', 'N58610', 'N60229', 'N60188', 'N60192', 'N60194', 'N60219', 'N60231']
    subjects = ['N58613']
    """
    subjects = ['N60167', 'N60133', 'N60200', 'N60131', 'N60139', 'N60163', 'N60159', 'N60157', 'N60127', 'N60161', 'N60169',
     'N60137', 'N58613', 'N60129']
    subjects = ['N58408', 'N58610', 'N58398', 'N58714', 'N58740', 'N58477', 'N58734', 'N58309', 'N58302', 'N58612', 'N58706', 'N58889', 'N58361', 'N58355', 'N59066', 'N58712', 'N58606', 'N58350', 'N58608', 'N58733', 'N58655', 'N58735', 'N58310', 'N58400', 'N58708', 'N58780', 'N58512', 'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58745', 'N58406', 'N58359', 'N58742', 'N58396', 'N58732', 'N58516', 'N58402']
    subjects = ['N58889', 'N59066']
    subjects = ['N57437', 'N57442', 'N57446', 'N57447', 'N57449', 'N57451', 'N57496', 'N57498', 'N57500', 'N57502', 'N57504', 'N57513', 'N57515', 'N57518', 'N57520', 'N57522', 'N57546', 'N57548', 'N57550', 'N57552', 'N57554', 'N57559', 'N57580', 'N57582', 'N57584', 'N57587', 'N57590', 'N57692', 'N57694', 'N57700', 'N57702', 'N57709']
    subjects = ['N57437', 'N57446','N57447','N57449','N57451','N57496','N57498','N57500','N57502','N57504','N57513','N57515','N57518','N57520','N57522','N57546','N57548','N57550','N57552','N57554','N57559','N57580','N57582','N57584','N57587','N57590','N57692','N57694','N57700','N57702','N57709','N58214','N58215','N58216','N58217' ,'N58218','N58219','N58221','N58222','N58223' ,'N58224','N58225','N58226','N58228','N58229','N58230','N58231','N58232','N58610','N58612','N58633','N58634','N58635','N58636','N58649','N58650','N58651','N58653','N58654','N58889','N59066','N59109']
    #subjects = ['N58613']
    subjects = ['M22090514']
    #subjects =['N58225', 'N58232', 'N58215', 'N58216', 'N58633', 'N58636', 'N58653', 'N58224', 'N58214', 'N58651', 'N58228', 'N58650', 'N58221', 'N58219', 'N58649', 'N58226', 'N58229', 'N58218', 'N58230', 'N58223', 'N58222', 'N58634', 'N58231', 'N58217', 'N58654', 'N58635']
    #subjects = ['N57442']
    #subjects = ['N58214',"N58215"]
    #subjects = ['N58613']
    #subjects = ['N58302', 'N58303']
    subjects = subjects[:]
    #removed_list = ['N58610', 'N58613', 'N58732']
    #removed_list = ['N58948','N59005','N58612','N58613','N59136','N59140','N58946','N59141','N58915','N58954']
    removed_list = ['N59109']
    for remove in removed_list:
        if remove in subjects:
            subjects.remove(remove)
    # subject 'N58610' 'N58612' 'N58813' retired, back on SAMBA_prep, to investigate

elif project == "AMD":

    SAMBA_mainpath = os.path.join('/Volumes/dusom_abadea_nas1/munin_js/VBM_backups')
    SAMBA_projectname = "VBM_19BrainChAMD01_IITmean_RPI_with_2yr"
    SAMBA_headfile = os.path.join(SAMBA_headfile_dir, "rja20_BrainChAMD.01_with_2yr_SAMBA_startup.headfile")
    gunniespath = "~/gunnies/"
    recenter = 0
    #SAMBA_prep_folder = os.path.join(SAMBA_mainpath, "whitson_symlink_pool_allfiles")
    SAMBA_prep_folder = os.path.join('/Volumes/dusom_mousebrains/All_Staff/Data/AMD/DWI')
    atlas_labels = os.path.join('/Volumes/Data/Badea/Lab/', "atlas","IITmean_RPI","IITmean_RPI_labels.nii.gz")

    DTC_DWI_folder = "DWI"
    DTC_labels_folder = "DWI"

    SAMBA_label_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-results", "connectomics")
    SAMBA_work_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-work")
    orient_string = os.path.join(SAMBA_prep_folder, "relative_orientation.txt")
    superpose = False
    copytype = "truecopy"
    overwrite = False

    subjects = ["H29056", "H26578", "H29060", "H26637", "H29264", "H26765", "H29225", "H26660", "H29304", "H26890", "H29556",
                "H26862", "H29410", "H26966", "H29403", "H26841", "H21593", "H27126", "H29618", "H27111", "H29627", "H27164",
                "H29502", "H27100", "H27381", "H21836", "H27391", "H21850", "H27495", "H21729", "H27488", "H21915", "H27682",
                "H21956", "H27686", "H22331", "H28208", "H21990", "H28955", "H29878", "H27719", "H22102", "H27841", "H22101",
                "H27842", "H22228", "H28029", "H22140", "H27852", "H22276", "H27999", "H22369", "H28115", "H22644", "H28308",
                "H22574", "H28377", "H22368", "H28325", "H22320", "H28182", "H22898", "H28748", "H22683", "H28373", "H22536",
                "H28433", "H22825", "H28662", "H22864", "H28698", "H23143", "H28861", "H23157", "H28820", "H23028", "H29002",
                "H23210", "H29020", "H23309", "H29161", "H26841", "H26862", "H26949", "H26966", "H27100", "H27126", "H27163",
                "H27246", "H27488", "H27682", "H27686", "H27719", "H27841", "H27842", "H27852", "H27869", "H27999", "H28029",
                "H28068", "H28208", "H28262", "H28325", "H28820", "H28856", "H28869", "H28955", "H29002", "H29044", "H29089",
                "H29127", "H29161", "H29242", "H29254", "H26578", "H26637", "H26660", "H26745", "H26765", "H26850", "H26880",
                "H26890", "H26958", "H26974", "H27017", "H27111", "H27164", "H27381", "H27391", "H27495", "H27610", "H27640",
                "H27680", "H27778", "H27982", "H28115", "H28308", "H28338", "H28373", "H28377", "H28433", "H28437", "H28463",
                "H28532", "H28662", "H28698", "H28748", "H28809", "H28857", "H28861", "H29013", "H29020", "H29025"]

    subjects = ['H21593', 'H21729', 'H21836', 'H21850', 'H21915', 'H21956', 'H21990', 'H22101', 'H22102', 'H22140', 'H22228', 'H22331', 'H22368', 'H22369', 'H22683', 'H22825', 'H22864', 'H22898', 'H23028', 'H23157', 'H23309', 'H26578', 'H26637', 'H26660', 'H26765', 'H26841', 'H26862', 'H26890', 'H26949', 'H26966', 'H27100', 'H27111', 'H27126', 'H27163', 'H27164', 'H27246', 'H27381', 'H27391', 'H27488', 'H27495', 'H27682', 'H27686', 'H27719', 'H27841', 'H27842', 'H27869', 'H28029', 'H28068', 'H28115', 'H28182', 'H28208', 'H28308', 'H28325', 'H28373', 'H28433', 'H28698', 'H28861', 'H28955', 'H29002', 'H29020', 'H29060', 'H29225', 'H29264', 'H29304', 'H29403', 'H29410', 'H29502', 'H29556', 'H29618', 'H29627', 'H29878', 'H22276', 'H22320', 'H22536', 'H22574', 'H22644', 'H23143', 'H23210', 'H26745', 'H26850', 'H26880', 'H26958', 'H26974', 'H27017', 'H27610', 'H27640', 'H27680', 'H27778', 'H27852', 'H27982', 'H27999', 'H28262', 'H28338', 'H28377', 'H28437', 'H28463', 'H28532', 'H28662', 'H28748', 'H28809', 'H28820', 'H28856', 'H28857', 'H28869', 'H29013', 'H29025', 'H29044', 'H29056', 'H29089', 'H29127', 'H29161', 'H29242', 'H29254']


elif project == "ADRC":

    SAMBA_mainpath = os.path.join(mainpath, "mouse")
    SAMBA_projectname = "VBM_23ADRC_IITmean_RPI"
    SAMBA_headfile = os.path.join(SAMBA_headfile_dir, "am983_SAMBA_ADRC_Jacques.headfile")
    gunniespath = "~/gunnies/"
    recenter = 0
    #SAMBA_prep_folder = os.path.join(SAMBA_mainpath, "whitson_symlink_pool_allfiles")
    SAMBA_prep_folder = os.path.join('/Volumes/Data/Badea/Lab/mouse/ADRC_jacques_pipeline/perm_files')
    atlas_labels = os.path.join('/Volumes/Data/Badea/Lab/', "atlas","IITmean_RPI","IITmean_RPI_labels.nii.gz")

    DTC_DWI_folder = "DWI"
    DTC_labels_folder = "DWI"

    SAMBA_label_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-results", "connectomics")
    SAMBA_work_folder = os.path.join(SAMBA_mainpath, SAMBA_projectname + "-work")
    orient_string = os.path.join(SAMBA_prep_folder, "relative_orientation.txt")
    superpose = False
    copytype = "truecopy"
    overwrite = False

    subjects_all = glob.glob(os.path.join(SAMBA_prep_folder,'*_fa.nii.gz'))
    subjects = []
    for subject in subjects_all:
        subject_name = os.path.basename(subject)
        subjects.append(subject_name[:8])
    subjects.sort()

else:
    raise Exception("Unknown project name")

remote=False
if remote:
    username, passwd = getfromfile(os.path.join(os.path.expanduser('~'),'remote_connect.rtf'))
else:
    username = None
    passwd = None


_, outpath, _, sftp = get_mainpaths(remote,project = project, username=username,password=passwd)

DTC_DWI_folder = os.path.join(outpath,DTC_DWI_folder)
DTC_labels_folder = os.path.join(outpath,DTC_labels_folder)
DTC_transforms = os.path.join(DTC_DWI_folder,'../Transforms')
mkcdir([outpath,DTC_DWI_folder,DTC_labels_folder,DTC_transforms],sftp)

print(subjects)

_, _, myiteration = get_info_SAMBA_headfile(SAMBA_headfile)
##### for ADRC for some reason????
################################

#subjects = ['S03394']
subjects_notdone = []
for subject in subjects:
    create_MDT_labels(subject, SAMBA_mainpath, SAMBA_projectname, atlas_labels, myiteration=myiteration,
                      overwrite=overwrite, verbose=verbose)
    labelspath_remote = os.path.join(DTC_labels_folder, f'{subject}_labels.nii.gz')
    """
    atlas_spe_labels = '/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/bundle_atlas/IIT_whitematter_RPI.nii.gz'
    label_name = 'IITmean_RPI'
    subject_notdone = create_backport_labels(subject, SAMBA_mainpath, SAMBA_projectname, SAMBA_prep_folder, atlas_spe_labels,
                                             headfile = SAMBA_headfile, overwrite=overwrite, verbose=verbose,
                                             identifier = identifier_SAMBA_folder,label_name = label_name)
    """
    subject_notdone = create_backport_labels(subject, SAMBA_mainpath, SAMBA_projectname, SAMBA_prep_folder, atlas_labels,
                                             headfile = SAMBA_headfile, overwrite=overwrite, verbose=verbose,
                                             identifier = identifier_SAMBA_folder)

    if subject_notdone is not None:
        subjects_notdone.append(subject_notdone)

print(f'subjects not done by SAMBA: {subjects_notdone}')
"""
txt2 = 'cp '
txt1=''
for subject in subjects_notdone:
    txt1= txt1 + (f'diffusion_prep_{subject} ')
    path1 = '/Volumes/Data/Badea/.snapshot/weekly.2023-02-05_0015/Lab/mouse/APOE_series/diffusion_prep_locale'
    file1 = os.path.join(path1,f'diffusion_prep_{subject}/Reg_{subject}nii4D.nii.gz')
    path2 = '/Volumes/Data/Badea/Lab/mouse/APOE_series/diffusion_prep_locale'
    file2 = os.path.join(path2,f'diffusion_prep_{subject}/Reg_{subject}nii4D.nii.gz')
    #txt2= txt2 + (f'diffusion_prep_{subject}/Reg_{subject}nii4D.nii.gz /Volumes/Data/Badea/Lab/mouse/APOE_series/diffusion_prep_locale/diffusion_prep_{subject}/')
    try:
        shutil.copy(file1, file2)
    except:
        print(f'cp {file1} {file2}')
    #shutil.copy(file1, file2)
    #print(f'cp {file1} {file2}')
"""

mkcdir([DTC_DWI_folder,DTC_labels_folder],sftp)

for filename in os.listdir(SAMBA_prep_folder):
    if any(x in filename for x in file_ids) and any(x in filename for x in subjects):
        filepath=os.path.join(SAMBA_prep_folder,filename)
        if 'N59010' in filename:
            print('hi')
        if Path(filepath).is_symlink():
            filepath=Path(filepath).resolve()
        filenewpath = os.path.join(DTC_DWI_folder, filename)
        if not os.path.isfile(filenewpath) or overwrite:
            if copytype=="shortcut":
                if remote:
                    raise Exception("Can't build shortcut to remote path")
                else:
                    buildlink(filepath, filenewpath)
            elif copytype=="truecopy":
                if remote:
                    if not overwrite:
                        try:
                            sftp.stat(filenewpath)
                            if verbose:
                                print(f'file at {filenewpath} exists')
                        except IOError:
                            if verbose:
                                print(f'copying file {filepath} to {filenewpath}')
                            sftp.put(filepath, filenewpath)
                    else:
                        if verbose:
                            print(f'copying file {filepath} to {filenewpath}')
                        try:
                            sftp.put(filepath, filenewpath)
                        except:
                            print('test')
                            #os.remove(filepath)
                else:
                    print(f'Copying {filepath} to {filenewpath}')
                    shutil.copy(filepath, filenewpath)
        else:
            if verbose:
                print(f'File {filenewpath} already exists')

reg_type = 'dwi'
template_type_prefix = os.path.basename(os.path.dirname(glob.glob(os.path.join(SAMBA_work_folder,"dwi","SyN*/"))[0]))

if myiteration == -1:
    template_runs = glob.glob((os.path.join(SAMBA_work_folder, reg_type, template_type_prefix, "*/")))
    for template_run in template_runs:
        if "NoNameYet" in template_run and template_run[-4:-2]=="_i":
            if int(template_run[-2])>myiteration:
                myiteration=int(template_run[-2])
                final_template_run=template_run
    if myiteration==-1:
        for template_run in template_runs:
            if "dwiMDT_Control_n72" in template_run and template_run[-4:-2]=="_i":
                if int(template_run[-2])>myiteration:
                    myiteration=int(template_run[-2])
                    final_template_run=template_run
    if myiteration == -1:
        raise Exception(f"Could not find template runs in {os.path.join(mainpath, f'{SAMBA_projectname}-work','dwi',template_type_prefix)}")

final_template_run = glob.glob(os.path.join(SAMBA_work_folder, reg_type, template_type_prefix, f"*i{myiteration}*/"))[0]

if project != "AMD":
    for subject in subjects:
        subjectpath = glob.glob(os.path.join(SAMBA_label_folder, f'{subject}/'))
        if np.size(subjectpath) == 1:
            subjectpath = subjectpath[0]
        elif np.size(subjectpath) > 1:
            raise Exception('Too many subject folders')
        else:
            subjectpath = SAMBA_label_folder

        labelspath = glob.glob(os.path.join(subjectpath, f'{subject}*{atlas_name}*_labels.nii*'))
        if np.size(labelspath) == 1:
            labelspath = labelspath[0]
        else:
            txt = f"Could not find file at {os.path.join(subjectpath, f'{subject}*{atlas_name}*_labels.nii*')}"
            warnings.warn(txt)
            continue
        newlabelspath = os.path.join(DTC_labels_folder,f'{subject}_labels.nii.gz')

        if not os.path.exists(newlabelspath) or overwrite:
            if remote:
                if not overwrite:
                    try:
                        sftp.stat(newlabelspath)
                        if verbose:
                            print(f'file at {newlabelspath} exists')
                    except IOError:
                        if verbose:
                            print(f'copying file {labelspath} to {newlabelspath}')
                        sftp.put(labelspath, newlabelspath)
                else:
                    sftp.put(labelspath, newlabelspath)
                    if verbose:
                        print(f'copying file {labelspath} to {newlabelspath}')

            else:
                shutil.copy(labelspath, newlabelspath)
                if verbose:
                    print(f'copying file {labelspath} to {newlabelspath}')
        else:
            if verbose:
                print(f"File already exists at {newlabelspath}")

        newlabelspath_ordered = newlabelspath.replace('_labels','_labels_lr_ordered')

        if not os.path.exists(newlabelspath_ordered) or overwrite:
            if remote:
                if not overwrite:
                    try:
                        sftp.stat(newlabelspath_ordered)
                        if verbose:
                            print(f'file at {newlabelspath_ordered} exists')
                    except IOError:
                        if verbose:
                            print(f'creating file {newlabelspath_ordered} from {labelspath}')
                        converter_lr, converter_comb, index_to_struct_lr, index_to_struct_comb = atlas_converter(
                            atlas_legends)
                        convert_labelmask(labelspath, converter_lr, atlas_outpath=newlabelspath_ordered, sftp=sftp)
                else:
                    if verbose:
                        print(f'creating file {newlabelspath_ordered} from {labelspath}')
                    converter_lr, converter_comb, index_to_struct_lr, index_to_struct_comb = atlas_converter(
                        atlas_legends)
                    convert_labelmask(labelspath, converter_lr, atlas_outpath=newlabelspath_ordered, sftp=sftp)

            else:
                converter_lr, converter_comb, index_to_struct_lr, index_to_struct_comb = atlas_converter(
                    atlas_legends)
                convert_labelmask(labelspath, converter_lr, atlas_outpath=newlabelspath_ordered)
                if verbose:
                    print(f'creating file {newlabelspath_ordered} from {labelspath}')
        else:
            if verbose:
                print(f"File already exists at {newlabelspath_ordered}")


elif project == "AMD":
    for subject in subjects:
        subjectpath = glob.glob(os.path.join(SAMBA_label_folder, f'{subject}/'))
        if np.size(subjectpath) == 1:
            subjectpath = subjectpath[0]
        elif np.size(subjectpath) > 1:
            raise Exception('Too many subject folders')
        else:
            subjectpath = SAMBA_label_folder

        labelspath = glob.glob(os.path.join(subjectpath, f'{subject}*_IITmean_RPI_labels.nii*'))
        coreg_prepro =  glob.glob(os.path.join(subjectpath, f'{subject}*_nii4D_masked_isotropic.nii*'))
        if np.size(labelspath) == 1:
            labelspath = labelspath[0]
        else:
            warnings.warn(f"Could not find file at {os.path.join(subjectpath, f'{subject}*_labels.nii*')}")
            continue

        if np.size(coreg_prepro) == 1:
            coreg_prepro = coreg_prepro[0]
        else:
            warnings.warn(f"Could not find file at {os.path.join(subjectpath, f'{subject}*_labels.nii*')}")
            continue

        newlabelspath = os.path.join(DTC_labels_folder, f'{subject}_labels.nii.gz')
        new_coregpath = os.path.join(DTC_labels_folder, f'{subject}_coreg_diff.nii.gz')

        if not os.path.exists(newlabelspath) or overwrite:
            if remote:
                if not overwrite:
                    try:
                        sftp.stat(newlabelspath)
                        if verbose:
                            print(f'file at {newlabelspath} exists')
                    except IOError:
                        if verbose:
                            print(f'copying file {labelspath} to {newlabelspath}')
                        sftp.put(labelspath, newlabelspath)
                else:
                    sftp.put(labelspath, newlabelspath)
                    if verbose:
                        print(f'copying file {labelspath} to {newlabelspath}')

            else:
                shutil.copy(labelspath, newlabelspath)
                if verbose:
                    print(f'copying file {labelspath} to {newlabelspath}')
        else:
            if verbose:
                print(f"File already exists at {newlabelspath}")

        if not os.path.exists(new_coregpath) or overwrite:
            if remote:
                if not overwrite:
                    try:
                        sftp.stat(new_coregpath)
                        if verbose:
                            print(f'file at {new_coregpath} exists')
                    except IOError:
                        if verbose:
                            print(f'copying file {coreg_prepro} to {new_coregpath}')
                        sftp.put(coreg_prepro, new_coregpath)
                else:
                    sftp.put(coreg_prepro, new_coregpath)
                    if verbose:
                        print(f'copying file {coreg_prepro} to {new_coregpath}')

            else:
                shutil.copy(coreg_prepro, new_coregpath)
                if verbose:
                    print(f'copying file {coreg_prepro} to {new_coregpath}')
        else:
            if verbose:
                print(f"File already exists at {new_coregpath}")

overwrite=False

for subject in subjects:
    trans = os.path.join(SAMBA_work_folder, "preprocess", "base_images", "translation_xforms",
                         f"{subject}_0DerivedInitialMovingTranslation.mat")
    rigid = os.path.join(SAMBA_work_folder, "dwi", f"{subject}_rigid.mat")
    affine = os.path.join(SAMBA_work_folder, "dwi", f"{subject}_affine.mat")
    runno_to_MDT = os.path.join(final_template_run, "reg_diffeo", f"{subject}_to_MDT_warp.nii.gz")

    burn_dir = os.path.join(SAMBA_mainpath, "burn_after_reading")
    affine_mat_path = os.path.join(burn_dir, f'{subject}_affine.txt')
    if not os.path.exists(affine_mat_path) or overwrite:
        cmd = f'ConvertTransformFile 3 {affine} {affine_mat_path} --matrix'
        os.system(cmd)

    transform_files = [trans, rigid, affine, affine_mat_path, runno_to_MDT]

    for filepath in transform_files:
        if os.path.exists(filepath):
            if Path(filepath).is_symlink():
                filepath=Path(filepath).resolve()
            filename = os.path.basename(filepath)
            filenewpath = os.path.join(DTC_transforms, filename)
            if not os.path.isfile(filenewpath) or overwrite:
                if copytype=="shortcut":
                    if remote:
                        raise Exception("Can't build shortcut to remote path")
                    else:
                        buildlink(filepath, filenewpath)
                        if verbose:
                            print(f'Built link for {filepath} at {filenewpath}')
                elif copytype=="truecopy":
                    if remote:
                        if not overwrite:
                            try:
                                sftp.stat(filenewpath)
                                if verbose:
                                    print(f'file at {filenewpath} exists')
                            except IOError:
                                if verbose:
                                    print(f'copying file {filepath} to {filenewpath}')
                                sftp.put(filepath, filenewpath)
                        else:
                            if verbose:
                                print(f'copying file {filepath} to {filenewpath}')
                            sftp.put(filepath, filenewpath)
                    else:
                        shutil.copy(filepath, filenewpath)
                        if verbose:
                            print(f'copying file {filepath} to {filenewpath}')
        else:
            print(f'Could not find {filepath}')

MDT_refs = ['fa', 'md', 'rd', 'ad', 'b0']
for subject in subjects:
    for MDT_ref in MDT_refs:
        filepath = os.path.join(final_template_run, 'reg_images',f'{subject}_{MDT_ref}_to_MDT.nii.gz')
        if not os.path.exists(filepath):
            txt = f'Could not find {filepath} in {final_template_run} for subject {subject} in project {project}'
            warnings.warn(txt)
        else:
            if Path(filepath).is_symlink():
                filepath=Path(filepath).resolve()
            filename = os.path.basename(filepath)
            filenewpath = os.path.join(DTC_labels_folder, filename)
            if not os.path.isfile(filenewpath) or overwrite:
                if copytype=="shortcut":
                    if remote:
                        raise Exception("Can't build shortcut to remote path")
                    else:
                        buildlink(filepath, filenewpath)
                        if verbose:
                            print(f'Built link for {filepath} at {filenewpath}')
                elif copytype=="truecopy":
                    if remote:
                        if not overwrite:
                            try:
                                sftp.stat(filenewpath)
                                if verbose:
                                    print(f'file at {filenewpath} exists')
                            except IOError:
                                if verbose:
                                    print(f'copying file {filepath} to {filenewpath}')
                                sftp.put(filepath, filenewpath)
                        else:
                            if verbose:
                                print(f'copying file {filepath} to {filenewpath}')
                            sftp.put(filepath, filenewpath)
                    else:
                        if verbose:
                            print(f'copying file {filepath} to {filenewpath}')
                        shutil.copy(filepath, filenewpath)

if remote:
    sftp.close()

"""
    APOE
    subjects = ['N57437', 'N57442', 'N57446', 'N57447','N57449','N57451','N57496','N57498','N57500','N57502','N57504', 'N57513',
                'N57515','N57518','N57520','N57522','N57546','N57548','N57550','N57552','N57554','N57559','N57580','N57582','N57584',
                'N57587','N57590','N57692','N57694','N57700','N57500','N57702','N57709',
                ,'N58214", ,'N58215", ,'N58216", ,'N58217", ,'N58218", ,'N58219", ,'N58221", ,'N58222", ,'N58223", ,'N58224",
                ,'N58225", ,'N58226", ,'N58228",
                ,'N58229", ,'N58230", ,'N58231", ,'N58232", ,'N58633", ,'N58634", ,'N58635", ,'N58636", ,'N58649", ,'N58650",
                ,'N58651", ,'N58653", ,'N58654",
                'N58408', 'N58398', 'N58714', 'N58740', 'N58477', 'N58734', 'N58309', 'N58792', 'N58302',
                'N58784', 'N58706', 'N58361', 'N58355', 'N58712', 'N58790', 'N58606', 'N58350', 'N58608',
                'N58779', 'N58500', 'N58604', 'N58749', 'N58510', 'N58394', 'N58346', 'N58344', 'N58788', 'N58305',
                'N58514', 'N58794', 'N58733', 'N58655', 'N58735', 'N58310', 'N58400', 'N58708', 'N58780', 'N58512',
                'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58745', 'N58406', 'N58359', 'N58742', 'N58396',
                'N58613', 'N58732', 'N58516', 'N58402']

"""
