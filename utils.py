import json
import os.path
import os.path as osp
from sklearn.model_selection import KFold
import glob
import numpy as np
import torch
import lifelines.utils.concordance as LUC
import random
import pickle

IGNORE ={'SY': [], #'19-12590','19-00754','18-01917','1316267','18-23299','19-23152'
        'GY': [], # '1941232','14-25684','16-829','17-4718','1915091','2016041','1906141','2018356'
        'ZY': [],
        'TCGA-UCEC':['TCGA-DF-A2KS'],
        'TCGA-KIRC':[],
        'TCGA-LUSC':[],
        'TCGA-LUAD':[],
        'TCGA-GBM':[]} #

# ZY:'0901608','0903875','0908064','0912803','0915370','0916767','0917436','0917669','0917747','1001723','1007863','1010052','1012389','1013992',
#                 '1014205','1015836','1100779','1100869','1101079','1102566','1103292','1104858','1109133','1109267','1110194','1111401','1114395','1117033',
#                 '1117472','1118403','1118862','1118874','1119371','1121275','1202116','1202898','1204003','1206519','1207326','1207352','1208098','1210590',
#                 '1213086','1214050','1219995','1222239','1222519','1301964','1305117','1306803'

# '1821945','2102347','1222519','1615277','2006273','1624652','1202116','1400527' ZY:'0901608','0903875','0908064','0912803','0915370','0916767','0917436','0917669','0917747','1001723','1007863','1010052','1012389','1013992',
#                 '1014205','1015836','1100779','1100869','1101079','1102566','1103292','1104858','1109133','1109267','1110194','1111401','1114395','1117033',
#                 '1117472','1118403','1118862','1118874','1119371','1121275','1202116','1202898','1204003','1206519','1207326','1207352','1208098','1210590',
#                 '1213086','1214050','1219995','1222239','1222519','1301964','1305117','1306803'

# '16-21833','17-08457','18-09327','18-18390','19-23653','20-06163','20-11266','20-13283','20-14768','1315421','1316959','1518224',
#                 '1401644','18-01917','18-23299','19-00754','19-12590','19-23152','19-28050','20-24013','1316267','16-19744','17-14963','17-22033',
#                 '17-26296','18-02928','18-06852','18-09680','18-14442','18-17218','18-26031','18-26284','19-03075','19-08033','19-09480','1404092', '1520286'

# '12-21078','14-25684','14-25832','16-829','16-10842','17-4718','17-17865','1906141','1915091','1937595','1941232','2016041',
#                '2018356','1810004','1910871','1912751','2000694'

# '1202116','1222519','1327596','1400527','1615277','1624652','1731488','1821945','2006273','2102347','1627627'
# SY_IGNORE = []
# '14-25684'
STAGE = {'TCGA-KIRC':{'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3},
         'TCGA-LUSC':{'Stage I': 0, 'Stage IA': 1, 'Stage IB': 2, 'Stage II': 3, 'Stage IIA': 4, 'Stage IIB': 5, 'Stage III': 6, 'Stage IIIA': 7, 'Stage IIIB': 8, 'Stage IV': 9},
         'TCGA-LUAD':{'Stage I': 0, 'Stage IA': 1, 'Stage IB': 2, 'Stage II': 3, 'Stage IIA': 4, 'Stage IIB': 5, 'Stage IIIA': 6, 'Stage IIIB': 7, 'Stage IV': 8},
         'TCGA-UCEC':{'Stage I': 0, 'Stage IA': 1, 'Stage IB': 2, 'Stage II': 3, 'Stage IIA': 4, 'Stage IIB': 5, 'Stage III': 6, 'Stage IIIA': 7, 'Stage IIIB': 8, 'Stage IV': 9}}
STAGE_T = {'TCGA-KIRC':{'T1':0,'T1a':1,'T1b':2,'T2':3,'T2a':4,'T2b':5,'T3':6,'T3a':7,'T3b':8,'T3c':9,'T4':10},
           'TCGA-LUSC':{'T1':0,'T1a':1,'T1b':2,'T2':3,'T2a':4,'T2b':5,'T3':6,'T3a':7,'T4':8},
           'TCGA-LUAD':{'T1':0,'T1a':1,'T1b':2,'T2':3,'T2a':4,'T2b':5,'T3':6,'T4':7, 'TX':8},
           'TCGA-UCEC':{'T1':0,'T1a':1,'T1b':2,'T2':3,'T2a':4,'T2b':5,'T3':6,'T4':7}}
STAGE_M = {'TCGA-KIRC':{'M0':0,'M1':1,'MX':2},
           'TCGA-LUSC':{'M0':0,'M1':1,'M1a':2,'M1b':3,'MX':4},
           'TCGA-LUAD':{'M0':0,'M1':1,'M1a':2,'M1b':3,'MX':4},
           'TCGA-UCEC':{'M0':0,'M1':1,'M1a':2,'M1b':3,'MX':4}}
STAGE_N = {'TCGA-KIRC':{'N0':0,'N1':1,'NX':2},
           'TCGA-LUSC':{'N0':0,'N1':1,'N2':2,'N3':3,'NX':4},
           'TCGA-LUAD':{'N0':0,'N1':1,'N2':2,'N3':3,'NX':4},
           'TCGA-UCEC':{'N0':0,'N1':1,'N2':2,'N3':3,'NX':4}}
GENDER = {'male':0,'female':1}

def get_WSI_sample_list(WSI_info_list,centers,patch_ft_dir,WSI_patch_coor_dir,drop_sample_num=None,multi_label=False):
    with open(WSI_info_list, 'r') as fp:
        lbls = json.load(fp)
    if WSI_patch_coor_dir is not None:
        all_coor_list = []
        if isinstance(centers,list):
            for center in centers:
                all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(center), '*_coors.pkl')))
        else:
            all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(centers), '*_coors.pkl')))
        for file in all_coor_list:
            with open(file,'rb') as f:
                data = pickle.load(f)
                if len(data)!=2000:
                   print(file.split('/')[-1].split('*_coors.pkl')[0])
        
        coor_dict = {}
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-10]
            return tmp_dir
        for _dir in all_coor_list:
            id = get_id(_dir)
            coor_dict[id] = _dir
    if patch_ft_dir is not None:
        all_ft_list = []
        if isinstance(centers,list):
            for center in centers:
                all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(center), '*_fts.npy')))
        else:
            all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(centers), '*_fts.npy')))
        

        
    all_dict = {}
    survival_time_max = 0
    survival_time_min = None
    none_list = []
    if 'TCGA' in WSI_info_list:
        for patient in lbls:
            # string = patient["exposures"][0]["submitter_id"][:12]
            if patient["exposures"][0]["submitter_id"][:12] not in IGNORE[centers[0]]:
                image_id = patient['diagnoses'][0]['submitter_id'].split('_')[0]
                all_dict[image_id] = {}
                if multi_label:
                    all_dict[image_id]['gender'] = GENDER[patient['demographic']['gender']]
                    if patient['demographic']["age_at_index"] is not None:
                        all_dict[image_id]['age'] = patient['demographic']["age_at_index"]
                    else:
                        del all_dict[image_id]
                        continue
                    if 'UCEC' not in WSI_info_list:
                        if "ajcc_pathologic_stage" in patient["diagnoses"][0].keys() and "ajcc_pathologic_t" in patient["diagnoses"][0].keys() \
                        and "ajcc_pathologic_m" in patient["diagnoses"][0].keys() and "ajcc_pathologic_n" in patient["diagnoses"][0].keys():
                            all_dict[image_id]['stage'] = STAGE[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_stage"]]
                            all_dict[image_id]['t'] = STAGE_T[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_t"]]
                            all_dict[image_id]['m'] = STAGE_M[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_m"]]
                            all_dict[image_id]['n'] = STAGE_N[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_n"]]
                        else:
                            del all_dict[image_id]
                            continue
                if 'days_to_death' in patient['demographic'].keys():
                    time = int(patient['demographic']['days_to_death'])
                    all_dict[image_id]['status'] = int(1)
                else:
                    try:
                        time = int(patient['diagnoses'][0]['days_to_last_follow_up'])
                        all_dict[image_id]['status'] = int(0)
                    except:
                        del all_dict[image_id]
                        continue
                all_dict[image_id]['survival_time'] = time
                

                #filter low survival time
                if time<=7:
                    del all_dict[image_id]
                    continue


                if str(image_id) in coor_dict.keys():
                        all_dict[image_id]['patch_coors'] = coor_dict[str(image_id)]
                else:
                    del all_dict[image_id]
                    none_list.append(image_id)
                    print('no coor_dir     '+image_id)
                survival_time_max = survival_time_max \
                        if survival_time_max > time else time
                if survival_time_min is None:
                    survival_time_min = time
                else:
                    survival_time_min = survival_time_min \
                        if survival_time_min < time else time
    else:
        for patient in lbls.keys():
            if str(lbls[patient]['pathology']) not in IGNORE[lbls[patient]['center']]:
                image_name = lbls[patient]['filename']
                image_id = str(lbls[patient]['pathology'])
                all_dict[image_id] = {}
                time = int(lbls[patient]['OS-time'])
                all_dict[image_id]['survival_time'] = time
                all_dict[image_id]['radiology'] = str(lbls[patient]['radiology']).replace(' ','')
                if str(image_id) in coor_dict.keys():
                    all_dict[image_id]['patch_coors'] = coor_dict[str(image_id)]
                all_dict[image_id]['status'] = int(lbls[patient]['OS'])
                all_dict[image_id]['filename'] = image_name

                if multi_label:
                    # all_dict[image_id]['Tumor size'] = float(lbls[patient]['Size'])
                    # all_dict[image_id]['WHO/ISUP grade'] = int(lbls[patient]['grading'])
                    # all_dict[image_id]['Necrosis_1'] = int(lbls[patient]['necrosis_1'])
                    all_dict[image_id]['T stage'] = int(lbls[patient]['T_stage'])
                    all_dict[image_id]['N stage'] = int(lbls[patient]['N_stage'])
                    all_dict[image_id]['M stage'] = int(lbls[patient]['M_stage'])
                    all_dict[image_id]['TNM stage'] = int(lbls[patient]['TNM_stage'])
                    # all_dict[image_id]['SSIGN_score'] = int(lbls[patient]['SSIGN_Score'])
                    # all_dict[image_id]['Tri_class'] = int(lbls[patient]['tri_class'])
                    # if centers=='GY':
                    #     all_dict[image_id]['Gender'] = lbls[patient]['gender']
                    #     all_dict[image_id]['Age'] = lbls[patient]['age']


                survival_time_max = survival_time_max \
                    if survival_time_max > time else time
                if survival_time_min is None:
                    survival_time_min = time
                else:
                    survival_time_min = survival_time_min \
                        if survival_time_min < time else time
            # else:
            #     print(lbls[patient]['pathology'])
    # a = all_dict['TCGA-BP-4797']
    if patch_ft_dir is not None:
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-8]
            return tmp_dir
        for _dir in all_ft_list:
            id = get_id(_dir)
            if id in all_dict.keys():
                all_dict[id]['ft_dir'] =_dir

    # if drop_sample_num is not None:
    #     sorted_st = sorted(all_dict.items(),key=lambda x: x[1]['survival_time'],reverse=True)
    #     count = 0
    #     for i in range(len(sorted_st)):
    #         if sorted_st[i][1]['status']==1:
    #             del all_dict[sorted_st[i][0]]
    #             count = count + 1
    #         if count == drop_sample_num:
    #             break
    print(sorted(none_list))
    if drop_sample_num is not None:
        sorted_st = sorted(all_dict.items(),key=lambda x: x[1]['survival_time'],reverse=True)
        count = 0
        for i in range(len(sorted_st)):
            if sorted_st[i][1]['status']==1:
                del all_dict[sorted_st[i][0]]
                count = count + 1
            if count == drop_sample_num[0]:
                break
        sorted_st = sorted(all_dict.items(),key=lambda x: x[1]['survival_time'])
        count = 0
        for i in range(len(sorted_st)):
            if sorted_st[i][1]['status']==0:
                del all_dict[sorted_st[i][0]]
                count = count + 1
            if count == drop_sample_num[1]:
                break

    # count = 0
    # while count < drop_sample_num:
    #     drop_id = random.randint(0,len(all_dict)-1)
    #     keys = list(all_dict.keys())
    #     if all_dict[keys[drop_id]]['status']==1:
    #         del all_dict[keys[drop_id]]
    #         count = count + 1
    #     if count == drop_sample_num:
    #         break

    return all_dict, survival_time_max, survival_time_min


def get_WSI_sample_list_nuclear(WSI_info_list, centers, patch_ft_dir, WSI_patch_coor_dir, nuclear_ft_dir,
                                drop_sample_num=None, multi_label=False):
    with open(WSI_info_list, 'r') as fp:
        lbls = json.load(fp)
    if WSI_patch_coor_dir is not None:
        all_coor_list = []
        if isinstance(centers, list):
            for center in centers:
                all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(center), '*_coors.pkl')))
        else:
            all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(centers), '*_coors.pkl')))
        for file in all_coor_list:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if len(data) != 2000:
                    print(file.split('/')[-1].split('*_coors.pkl')[0])

        coor_dict = {}

        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-10]
            return tmp_dir

        for _dir in all_coor_list:
            id = get_id(_dir)
            coor_dict[id] = _dir
    if patch_ft_dir is not None:
        all_ft_list = []
        if isinstance(centers, list):
            for center in centers:
                all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(center), '*_fts.npy')))
        else:
            all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(centers), '*_fts.npy')))

    if nuclear_ft_dir is not None:
        all_nuclear_list = []
        if isinstance(centers, list):
            for center in centers:
                all_nuclear_list.extend(glob.glob(osp.join(nuclear_ft_dir.format(center), '*_nuc.npy')))
        else:
            all_nuclear_list.extend(glob.glob(osp.join(nuclear_ft_dir.format(centers), '*_nuc.npy')))

    all_dict = {}
    survival_time_max = 0
    survival_time_min = None
    none_list = []

    if 'TCGA-COAD' in WSI_info_list:
        for patient in lbls:
            image_id = patient['submitter_id']
            all_dict[image_id] = {}
            if 'days_to_death' in patient.keys():
                time = int(patient['days_to_death'])
                all_dict[image_id]['status'] = int(1)
            else:
                try:
                    time = int(patient['days_to_last_follow_up'])
                    all_dict[image_id]['status'] = int(0)
                except:
                    del all_dict[image_id]
                    continue
            all_dict[image_id]['survival_time'] = time

            if str(image_id) in coor_dict.keys():
                all_dict[image_id]['patch_coors'] = coor_dict[str(image_id)]
            else:
                del all_dict[image_id]
                none_list.append(image_id)
                print('no coor_dir     ' + image_id)
            survival_time_max = survival_time_max \
                if survival_time_max > time else time
            if survival_time_min is None:
                survival_time_min = time
            else:
                survival_time_min = survival_time_min \
                    if survival_time_min < time else time

    elif 'TCGA' in WSI_info_list:
        for patient in lbls:
            # string = patient["exposures"][0]["submitter_id"][:12]
            if patient["exposures"][0]["submitter_id"][:12] not in IGNORE[centers[0]]:
                image_id = patient['diagnoses'][0]['submitter_id'].split('_')[0]
                all_dict[image_id] = {}
                if multi_label:
                    all_dict[image_id]['gender'] = GENDER[patient['demographic']['gender']]
                    if patient['demographic']["age_at_index"] is not None:
                        all_dict[image_id]['age'] = patient['demographic']["age_at_index"]
                    else:
                        del all_dict[image_id]
                        continue
                    if 'UCEC' not in WSI_info_list:
                        if "ajcc_pathologic_stage" in patient["diagnoses"][0].keys() and "ajcc_pathologic_t" in \
                                patient["diagnoses"][0].keys() \
                                and "ajcc_pathologic_m" in patient["diagnoses"][0].keys() and "ajcc_pathologic_n" in \
                                patient["diagnoses"][0].keys():
                            all_dict[image_id]['stage'] = STAGE[centers[0]][
                                patient["diagnoses"][0]["ajcc_pathologic_stage"]]
                            all_dict[image_id]['t'] = STAGE_T[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_t"]]
                            all_dict[image_id]['m'] = STAGE_M[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_m"]]
                            all_dict[image_id]['n'] = STAGE_N[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_n"]]
                        else:
                            del all_dict[image_id]
                            continue
                if 'days_to_death' in patient['demographic'].keys():
                    time = int(patient['demographic']['days_to_death'])
                    all_dict[image_id]['status'] = int(1)
                else:
                    try:
                        time = int(patient['diagnoses'][0]['days_to_last_follow_up'])
                        all_dict[image_id]['status'] = int(0)
                    except:
                        del all_dict[image_id]
                        continue
                all_dict[image_id]['survival_time'] = time

                # filter low survival time
                if time <= 7:
                    del all_dict[image_id]
                    continue

                if str(image_id) in coor_dict.keys():
                    all_dict[image_id]['patch_coors'] = coor_dict[str(image_id)]
                else:
                    del all_dict[image_id]
                    none_list.append(image_id)
                    print('no coor_dir     ' + image_id)
                survival_time_max = survival_time_max \
                    if survival_time_max > time else time
                if survival_time_min is None:
                    survival_time_min = time
                else:
                    survival_time_min = survival_time_min \
                        if survival_time_min < time else time
    else:
        for patient in lbls.keys():
            if str(lbls[patient]['pathology']) not in IGNORE[lbls[patient]['center']]:
                image_name = lbls[patient]['filename']
                image_id = str(lbls[patient]['pathology'])
                all_dict[image_id] = {}
                time = int(lbls[patient]['OS-time'])
                all_dict[image_id]['survival_time'] = time
                all_dict[image_id]['radiology'] = str(lbls[patient]['radiology']).replace(' ', '')
                if str(image_id) in coor_dict.keys():
                    all_dict[image_id]['patch_coors'] = coor_dict[str(image_id)]
                all_dict[image_id]['status'] = int(lbls[patient]['OS'])
                all_dict[image_id]['filename'] = image_name

                if multi_label:
                    # all_dict[image_id]['Tumor size'] = float(lbls[patient]['Size'])
                    # all_dict[image_id]['WHO/ISUP grade'] = int(lbls[patient]['grading'])
                    # all_dict[image_id]['Necrosis_1'] = int(lbls[patient]['necrosis_1'])
                    all_dict[image_id]['T stage'] = int(lbls[patient]['T_stage'])
                    all_dict[image_id]['N stage'] = int(lbls[patient]['N_stage'])
                    all_dict[image_id]['M stage'] = int(lbls[patient]['M_stage'])
                    all_dict[image_id]['TNM stage'] = int(lbls[patient]['TNM_stage'])
                    # all_dict[image_id]['SSIGN_score'] = int(lbls[patient]['SSIGN_Score'])
                    # all_dict[image_id]['Tri_class'] = int(lbls[patient]['tri_class'])
                    # if centers=='GY':
                    #     all_dict[image_id]['Gender'] = lbls[patient]['gender']
                    #     all_dict[image_id]['Age'] = lbls[patient]['age']

                survival_time_max = survival_time_max \
                    if survival_time_max > time else time
                if survival_time_min is None:
                    survival_time_min = time
                else:
                    survival_time_min = survival_time_min \
                        if survival_time_min < time else time
            # else:
            #     print(lbls[patient]['pathology'])
    # a = all_dict['TCGA-BP-4797']
    if patch_ft_dir is not None:
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-8]
            return tmp_dir

        for _dir in all_ft_list:
            id = get_id(_dir)
            if id in all_dict.keys():
                all_dict[id]['ft_dir'] = _dir

        for _dir in all_nuclear_list:
            id = get_id(_dir)
            if id in all_dict.keys():
                all_dict[id]['nuclear_ft_dir'] = _dir

    for _key in list(all_dict.keys()):
        if 'ft_dir' not in all_dict[_key].keys():
            del all_dict[_key]
            continue
        if 'nuclear_ft_dir' not in all_dict[_key].keys():
            del all_dict[_key]
            continue

    # if drop_sample_num is not None:
    #     sorted_st = sorted(all_dict.items(),key=lambda x: x[1]['survival_time'],reverse=True)
    #     count = 0
    #     for i in range(len(sorted_st)):
    #         if sorted_st[i][1]['status']==1:
    #             del all_dict[sorted_st[i][0]]
    #             count = count + 1
    #         if count == drop_sample_num:
    #             break
    if drop_sample_num is not None:
        sorted_st = sorted(all_dict.items(), key=lambda x: x[1]['survival_time'], reverse=True)
        count = 0
        for i in range(len(sorted_st)):
            if sorted_st[i][1]['status'] == 1:
                del all_dict[sorted_st[i][0]]
                count = count + 1
            if count == drop_sample_num[0]:
                break
        sorted_st = sorted(all_dict.items(), key=lambda x: x[1]['survival_time'])
        count = 0
        for i in range(len(sorted_st)):
            if sorted_st[i][1]['status'] == 0:
                del all_dict[sorted_st[i][0]]
                count = count + 1
            if count == drop_sample_num[1]:
                break

    # count = 0
    # while count < drop_sample_num:
    #     drop_id = random.randint(0,len(all_dict)-1)
    #     keys = list(all_dict.keys())
    #     if all_dict[keys[drop_id]]['status']==1:
    #         del all_dict[keys[drop_id]]
    #         count = count + 1
    #     if count == drop_sample_num:
    #         break

    return all_dict, survival_time_max, survival_time_min


def get_WSI_sample_list_ct(WSI_info_list, centers, patch_ft_dir, WSI_patch_coor_dir, CT_2d_feature_file=None,
                           drop_sample_num=None, multi_label=False):
    with open(WSI_info_list, 'r') as fp:
        lbls = json.load(fp)
    if WSI_patch_coor_dir is not None:
        all_coor_list = []
        if isinstance(centers, list):
            for center in centers:
                all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(center), '*_coors.pkl')))
        else:
            all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(centers), '*_coors.pkl')))
        for file in all_coor_list:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if len(data) != 2000:
                    print(file.split('/')[-1].split('*_coors.pkl')[0])

        coor_dict = {}

        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-10]
            return tmp_dir

        for _dir in all_coor_list:
            id = get_id(_dir)
            coor_dict[id] = _dir
    if patch_ft_dir is not None:
        all_ft_list = []
        if isinstance(centers, list):
            for center in centers:
                all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(center), '*_fts.npy')))
        else:
            all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(centers), '*_fts.npy')))

    all_dict = {}
    survival_time_max = 0
    survival_time_min = None
    none_list = []
    if 'TCGA' in WSI_info_list:
        for patient in lbls:
            # string = patient["exposures"][0]["submitter_id"][:12]
            if patient["exposures"][0]["submitter_id"][:12] not in IGNORE[centers[0]]:
                image_id = patient['diagnoses'][0]['submitter_id'].split('_')[0]
                all_dict[image_id] = {}
                if multi_label:
                    all_dict[image_id]['gender'] = GENDER[patient['demographic']['gender']]
                    if patient['demographic']["age_at_index"] is not None:
                        all_dict[image_id]['age'] = patient['demographic']["age_at_index"]
                    else:
                        del all_dict[image_id]
                        continue
                    if 'UCEC' not in WSI_info_list:
                        if "ajcc_pathologic_stage" in patient["diagnoses"][0].keys() and "ajcc_pathologic_t" in \
                                patient["diagnoses"][0].keys() \
                                and "ajcc_pathologic_m" in patient["diagnoses"][0].keys() and "ajcc_pathologic_n" in \
                                patient["diagnoses"][0].keys():
                            all_dict[image_id]['stage'] = STAGE[centers[0]][
                                patient["diagnoses"][0]["ajcc_pathologic_stage"]]
                            all_dict[image_id]['t'] = STAGE_T[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_t"]]
                            all_dict[image_id]['m'] = STAGE_M[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_m"]]
                            all_dict[image_id]['n'] = STAGE_N[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_n"]]
                        else:
                            del all_dict[image_id]
                            continue
                if 'days_to_death' in patient['demographic'].keys():
                    time = int(patient['demographic']['days_to_death'])
                    all_dict[image_id]['status'] = int(1)
                else:
                    try:
                        time = int(patient['diagnoses'][0]['days_to_last_follow_up'])
                        all_dict[image_id]['status'] = int(0)
                    except:
                        del all_dict[image_id]
                        continue
                all_dict[image_id]['survival_time'] = time

                # filter low survival time
                if time <= 7:
                    del all_dict[image_id]
                    continue

                if str(image_id) in coor_dict.keys():
                    all_dict[image_id]['patch_coors'] = coor_dict[str(image_id)]
                else:
                    del all_dict[image_id]
                    none_list.append(image_id)
                    print('no coor_dir     ' + image_id)
                survival_time_max = survival_time_max \
                    if survival_time_max > time else time
                if survival_time_min is None:
                    survival_time_min = time
                else:
                    survival_time_min = survival_time_min \
                        if survival_time_min < time else time
    else:
        for patient in lbls.keys():
            if str(lbls[patient]['pathology']) not in IGNORE[lbls[patient]['center']]:
                image_name = lbls[patient]['filename']
                image_id = str(lbls[patient]['pathology'])
                all_dict[image_id] = {}
                time = int(lbls[patient]['OS-time'])
                all_dict[image_id]['survival_time'] = time
                all_dict[image_id]['radiology'] = str(lbls[patient]['radiology']).replace(' ', '')
                if str(image_id) in coor_dict.keys():
                    all_dict[image_id]['patch_coors'] = coor_dict[str(image_id)]
                all_dict[image_id]['status'] = int(lbls[patient]['OS'])
                all_dict[image_id]['filename'] = image_name

                if multi_label:
                    # all_dict[image_id]['Tumor size'] = float(lbls[patient]['Size'])
                    # all_dict[image_id]['WHO/ISUP grade'] = int(lbls[patient]['grading'])
                    # all_dict[image_id]['Necrosis_1'] = int(lbls[patient]['necrosis_1'])
                    all_dict[image_id]['T stage'] = int(lbls[patient]['T_stage'])
                    all_dict[image_id]['N stage'] = int(lbls[patient]['N_stage'])
                    all_dict[image_id]['M stage'] = int(lbls[patient]['M_stage'])
                    all_dict[image_id]['TNM stage'] = int(lbls[patient]['TNM_stage'])
                    # all_dict[image_id]['SSIGN_score'] = int(lbls[patient]['SSIGN_Score'])
                    # all_dict[image_id]['Tri_class'] = int(lbls[patient]['tri_class'])
                    # if centers=='GY':
                    #     all_dict[image_id]['Gender'] = lbls[patient]['gender']
                    #     all_dict[image_id]['Age'] = lbls[patient]['age']

                survival_time_max = survival_time_max \
                    if survival_time_max > time else time
                if survival_time_min is None:
                    survival_time_min = time
                else:
                    survival_time_min = survival_time_min \
                        if survival_time_min < time else time
            # else:
            #     print(lbls[patient]['pathology'])
    # a = all_dict['TCGA-BP-4797']
    if patch_ft_dir is not None:
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-8]
            return tmp_dir

        for _dir in all_ft_list:
            id = get_id(_dir)
            if id in all_dict.keys():
                all_dict[id]['ft_dir'] = _dir

    # if drop_sample_num is not None:
    #     sorted_st = sorted(all_dict.items(),key=lambda x: x[1]['survival_time'],reverse=True)
    #     count = 0
    #     for i in range(len(sorted_st)):
    #         if sorted_st[i][1]['status']==1:
    #             del all_dict[sorted_st[i][0]]
    #             count = count + 1
    #         if count == drop_sample_num:
    #             break
    print(sorted(none_list))
    if drop_sample_num is not None:
        sorted_st = sorted(all_dict.items(), key=lambda x: x[1]['survival_time'], reverse=True)
        count = 0
        for i in range(len(sorted_st)):
            if sorted_st[i][1]['status'] == 1:
                del all_dict[sorted_st[i][0]]
                count = count + 1
            if count == drop_sample_num[0]:
                break
        sorted_st = sorted(all_dict.items(), key=lambda x: x[1]['survival_time'])
        count = 0
        for i in range(len(sorted_st)):
            if sorted_st[i][1]['status'] == 0:
                del all_dict[sorted_st[i][0]]
                count = count + 1
            if count == drop_sample_num[1]:
                break

    if CT_2d_feature_file is not None:
        with open(CT_2d_feature_file, 'rb') as f:
            ct_2d_features = pickle.load(f)
    for _key in all_dict.keys():
        if CT_2d_feature_file is not None:
            if _key in ct_2d_features['axial'].keys():
                all_dict[_key]['axial'] = ct_2d_features['axial'][_key]
            if _key in ct_2d_features['sagittal'].keys():
                all_dict[_key]['sagittal'] = ct_2d_features['sagittal'][_key]
            if _key in ct_2d_features['coronal'].keys():
                all_dict[_key]['coronal'] = ct_2d_features['coronal'][_key]

    key_list = list(all_dict.keys())
    for k in key_list:
        if ('axial' not in all_dict[k].keys() or 'sagittal' not in all_dict[k].keys()
                or 'coronal' not in all_dict[k].keys()):
            del all_dict[k]
            print('no CT feature: {}'.format(k))

    # count = 0
    # while count < drop_sample_num:
    #     drop_id = random.randint(0,len(all_dict)-1)
    #     keys = list(all_dict.keys())
    #     if all_dict[keys[drop_id]]['status']==1:
    #         del all_dict[keys[drop_id]]
    #         count = count + 1
    #     if count == drop_sample_num:
    #         break

    return all_dict, survival_time_max, survival_time_min

def get_WSI_sample_list_tiantan(WSI_info_list,centers,patch_ft_dir,WSI_patch_coor_dir,drop_sample_num=None,multi_label=False):
    with open(WSI_info_list, 'r') as fp:
        lbls = json.load(fp)
    if WSI_patch_coor_dir is not None:
        all_coor_list = []
        if isinstance(centers, list):
            for center in centers:
                all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(center), '*_coors.pkl')))
        else:
            all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(centers), '*_coors.pkl')))

        coor_dict = {}

        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1].split('_coors.pkl')[0]
            return tmp_dir

        for _dir in all_coor_list:
            id = get_id(_dir)
            coor_dict[id] = _dir

    if patch_ft_dir is not None:
        all_ft_list = []
        if isinstance(centers, list):
            for center in centers:
                all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(center), '*_fts.npy')))
        else:
            all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(centers), '*_fts.npy')))

    all_dict = {}
    survival_time_max = 0
    survival_time_min = None

    for image_id in sorted(lbls.keys()):
        all_dict[image_id] = {}
        all_dict[image_id]['survival_time'] = int(lbls[image_id]['survival_time'])
        all_dict[image_id]['status'] = int(lbls[image_id]['status'])

        survival_time_max = max(survival_time_max, all_dict[image_id]['survival_time'])
        survival_time_min = min(survival_time_min, all_dict[image_id]['survival_time']) \
            if survival_time_min is not None else all_dict[image_id]['survival_time']

        all_dict[image_id]['patch_coors'] = coor_dict[image_id]
    if patch_ft_dir is not None:
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1].split('_fts.npy')[0]
            return tmp_dir

        for _dir in all_ft_list:
            id = get_id(_dir)
            if id in all_dict.keys():
                all_dict[id]['ft_dir'] =_dir
    return all_dict, survival_time_max, survival_time_min


def get_n_fold_data_list(data_dict,n_fold,random_seed):
    censored_keys = []
    uncensored_keys = []
    for key in data_dict.keys():
        if data_dict[key]['status'] == 1:
            uncensored_keys.append(key)
        else:
            censored_keys.append(key)
    print("censored length {}".format(len(censored_keys)))
    print("uncensored length {}".format(len(uncensored_keys)))

    n_fold_uncensored_train_list = []
    n_fold_uncensored_val_list = []
    n_fold_censored_train_list = []
    n_fold_censored_val_list = []
    n_fold_train_list = []
    n_fold_val_list = []
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed) #random_seed
    for train_idx, val_idx in kf.split(uncensored_keys):
        train_keys = [uncensored_keys[i] for i in train_idx]
        val_keys = [uncensored_keys[i] for i in val_idx]

        train_data_dict = {key: data_dict[key] for key in train_keys}
        val_data_dict = {key: data_dict[key] for key in val_keys}
        n_fold_uncensored_train_list.append(train_data_dict)
        n_fold_uncensored_val_list.append(val_data_dict)

    idx = 0

    for train_idx, val_idx in kf.split(censored_keys):
        train_keys = [censored_keys[i] for i in train_idx]
        val_keys = [censored_keys[i] for i in val_idx]

        train_data_dict = {key: data_dict[key] for key in train_keys}
        val_data_dict = {key: data_dict[key] for key in val_keys}
        n_fold_censored_train_list.append(train_data_dict)
        n_fold_censored_val_list.append(val_data_dict)

    #     idx += 1
    #
    #     with open(os.path.join(osp.join('/home2/yanjielong/IIHGC/dataset/WSI_info_list', 'KIRC-split.txt')), 'a') as f:
    #         f.write(f"Fold {idx} train_keys: {train_keys}")
    #         f.write('\n')
    #         f.write(f"Fold {idx} val_keys: {val_keys}")
    #         f.write('\n')
    #
    #     print(f"Fold {idx} train_keys: {train_keys}")
    #     print(f"Fold {idx} val_keys: {val_keys}")
    #
    # exit(0)

    for i in range(n_fold):
        n_fold_train_list.append(dict(n_fold_censored_train_list[i],**n_fold_uncensored_train_list[i]))
        n_fold_val_list.append(dict(n_fold_censored_val_list[i],**n_fold_uncensored_val_list[i]))


    return n_fold_train_list, n_fold_val_list


def get_n_fold_data_list_tiantan_jt_split(data_dict, n_fold, random_seed):
    split_file = '/home2/yanjielong/IIHGC/dataset/WSI_info_list/tiantan_jt_split.json'
    with open(split_file, 'r') as fp:
        split_dict = json.load(fp)

    n_fold_train_list = []
    n_fold_val_list = []
    n_fold_train_key_list = []
    n_fold_val_key_list = []
    for i in range(n_fold):
        n_fold_train_key_list.append([])
        n_fold_val_key_list.append([])
        for key in data_dict.keys():
            if int(key) in split_dict[str(i)]:
                n_fold_val_key_list[i].append(key)
            else:
                n_fold_train_key_list[i].append(key)


    for i in range(n_fold):
        n_fold_train_list.append({key: data_dict[key] for key in n_fold_train_key_list[i]})
        n_fold_val_list.append({key: data_dict[key] for key in n_fold_val_key_list[i]})

    print()

    return n_fold_train_list, n_fold_val_list

def sort_survival_time(gt_survival_time,pre_risk,censore, output_fts,patch_ft=None,coors=None):
    ix = torch.argsort(gt_survival_time, dim= 0, descending=True)#
    gt_survival_time = gt_survival_time[ix]
    pre_risk = pre_risk[ix]
    censore = censore[ix]
    output_fts = output_fts[ix]
    if patch_ft is not None:
        patch_ft = patch_ft[ix]
        coors = coors[ix]
        return gt_survival_time,pre_risk,censore,output_fts,patch_ft,coors
    return gt_survival_time,pre_risk,censore,output_fts

def accuracytest(survivals, risk, censors):
    survlist = []
    risklist = []
    censorlist = []

    for riskval in risk:
        # riskval = -riskval
        risklist.append(riskval.cpu().detach().item())

    for censorval in censors:
        censorlist.append(censorval.cpu().detach().item())

    for surval in survivals:
        # surval = -surval
        survlist.append(surval.cpu().detach().item())

    C_value = LUC.concordance_index(survlist, risklist, censorlist)

    return C_value

