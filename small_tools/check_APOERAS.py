import os

#full SAMBA set
subjects=['N57437','N57442','N57446','N57447','N57449','N57451','N57496','N57498','N57500','N57502','N57504','N57513','N57515','N57518','N57520','N57522','N57546','N57548','N57550','N57552','N57554','N57559','N57580','N57582','N57584','N57587','N57590','N57692','N57694','N57700','N57702','N57709','N58302','N58303','N58305','N58309','N58310','N58344','N58346','N58350','N58355','N58359','N58361','N58394','N58396','N58398','N58400','N58402','N58404','N58406','N58408','N58477','N58500','N58510','N58512','N58514','N58516','N58604','N58606','N58608','N58611','N58613','N58706','N58708','N58712','N58214','N58215','N58216','N58217','N58218','N58219','N58221','N58222','N58223','N58224','N58225','N58226','N58228','N58229','N58230','N58231','N58232','N58633','N58634','N58635','N58636','N58649','N58650','N58651','N58653','N58654','N58714','N58740','N58734','N58792','N58784','N58790','N58779','N58749','N58788','N58794','N58733','N58655','N58735','N58780','N58747','N58751','N58745','N58742','N58732','N58935','N59003','N58819','N58909','N58919','N58889','N59010','N58859','N58917','N58815','N58997','N58999','N58881','N58853','N58995','N58877','N58883','N58885','N58906','N58821','N58855','N58861','N58857','N58851','N58887','N58879','N58829','N58913','N58831','N58941','N58813','N58952','N59022','N59026','N59033','N59035','N59039','N59041','N59065','N59066','N59072','N59076','N59078','N59080','N59097','N59099','N59109','N59116','N59118','N59120','N58612','N59136','N59140','N58946','N59141','N58915','N59005','N58954','N58948','N58610','N60103','N60062','N60101','N60056','N60088','N60064','N60070','N60093','N60097','N60095','N60068','N60072','N60058','N60092','N60060','N60188','N60190','N60192','N60194','N60198','N60219','N60221','N60223','N60225','N60229','N60231']

#18abb11 set
#subjects = ['N58408', 'N59072', 'N58610', 'N58398', 'N58935', 'N60219', 'N58714', 'N60221', 'N58740', 'N58477', 'N59003', 'N58734', 'N60229', 'N58309', 'N58792', 'N58819', 'N58302', 'N59078', 'N59116', 'N58909', 'N58612', 'N59136', 'N59140', 'N58784', 'N58919', 'N58706', 'N60103', 'N60167', 'N60198', 'N60062', 'N58889', 'N58361', 'N58355', 'N60101', 'N59066', 'N58712', 'N58790', 'N59010', 'N58859', 'N58946', 'N58917', 'N58606', 'N60133', 'N58815', 'N59118', 'N60056', 'N60200', 'N60131', 'N60088', 'N58997', 'N58350', 'N59022', 'N58999', 'N59141', 'N60188', 'N58881', 'N59026', 'N58608', 'N58853', 'N58779', 'N60139', 'N58995', 'N58500', 'N60163', 'N58604', 'N58749', 'N58877', 'N58883', 'N58915', 'N60064', 'N59109', 'N59120', 'N60231', 'N58510', 'N58885', 'N58906', 'N60190', 'N59065', 'N58394', 'N58821', 'N58855', 'N58346', 'N58861', 'N59005', 'N58344', 'N60070', 'N58954', 'N59099', 'N60093', 'N58857', 'N60159', 'N58788', 'N58305', 'N58514', 'N58851', 'N59076', 'N59097', 'N58794', 'N58733', 'N58655', 'N58887', 'N60223', 'N58735', 'N58310', 'N60097', 'N60095', 'N59035', 'N58879', 'N58400', 'N59041', 'N60068', 'N58952', 'N58708', 'N60157', 'N58780', 'N58512', 'N58747', 'N58303', 'N58404', 'N58751', 'N58611', 'N58829', 'N60127', 'N60161', 'N58913', 'N60072', 'N60192', 'N60169', 'N58745', 'N58831', 'N58406', 'N60137', 'N58359', 'N58742', 'N58396', 'N58613', 'N58948', 'N58941', 'N59033', 'N58732', 'N60194', 'N60058', 'N58516', 'N59080', 'N60129', 'N60092', 'N58813', 'N60060', 'N59039', 'N60225', 'N58402']

DWI_folder = '/Volumes/Data/Badea/Lab/APOE/DWI_allsubj/'

missing_subj = []
done_subj = []
for subject in subjects:
    coreg_subj = os.path.join(DWI_folder,f'{subject}_subjspace_coreg.nii.gz')
    labels = os.path.join(DWI_folder,f'{subject}_labels.nii.gz')
    bvals = os.path.join(DWI_folder,f'{subject}_bvals.txt')
    bvecs = os.path.join(DWI_folder,f'{subject}_bvecs.txt')
    dwi = os.path.join(DWI_folder,f'{subject}_subjspace_dwi.nii.gz')
    if not os.path.exists(coreg_subj) or not os.path.exists(bvals) or not os.path.exists(bvecs) or not os.path.exists(dwi):
        missing_subj.append(subject)
    else:
        done_subj.append(subject)
print(missing_subj)

subjects = done_subj

DWI_RAS_folder = '/Volumes/Data/Badea/Lab/APOE/DWI_allsubj_RAS/'

missing_subj = []
done_subj = []
for subject in subjects:
    coreg_subj = os.path.join(DWI_RAS_folder,f'{subject}_coreg_RAS.nii.gz')
    labels = os.path.join(DWI_RAS_folder,f'{subject}_labels_lr_ordered_RAS.nii.gz')
    bvals = os.path.join(DWI_RAS_folder,f'{subject}_bvals.txt')
    bvecs = os.path.join(DWI_RAS_folder,f'{subject}_bvecs.txt')
    dwi = os.path.join(DWI_RAS_folder,f'{subject}_dwi_RAS.nii.gz')
    if not os.path.exists(coreg_subj) or not os.path.exists(labels) or not os.path.exists(bvals) or not os.path.exists(bvecs) or not os.path.exists(dwi):
        missing_subj.append(subject)
    else:
        done_subj.append(subject)
print(missing_subj)
