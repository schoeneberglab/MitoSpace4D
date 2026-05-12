import argparse
import json
import pubchempy as pcp
import os.path as osp

dose_dict = {'cccp': '10 um',
             'valinomycin': '100 nM',
             'control': '1% DMSO',
             'h2o2': '500 uM',
             'rotenone': '10 uM',
             'mitoq': '10 uM',
             'nocodazole': '10 uM',
             'colchicine': '10 uM',
             'myls22': '100 uM',
             'dnp': '100 uM',
             'cytochalasind': '10uM',
             'nigericin': '10 uM',
             'oligomycin': '50 uM',
             'paraquat': '10 mM',
             'resveratrol': '100 uM',
             'tbhp': '500 uM',
             'lantrunculinb': '10 uM',
             'cisplatin': '100 uM',
             'mfi8': '50 uM',
             'azide': '10 mM',
             'lonidamine': '100 uM',
             'p110': '10 uM',
             'mitomycinc': '100 uM',
             'antimycina': '100 uM',
             'mdivi1': '100 uM',
             'tiron': '10 mM'
             }


if __name__ == '__main__':
    proj_dir = "/home/dhruvagarwal/projects/MitoSpace4D/"
    save_dir = osp.join(proj_dir, "runs", "lightning_logs", 'resnetbilstm_encoded_normal')
    metadata = json.load(open(f"{save_dir}/metadata.json"))

    # parse the whole metadata and take out all the 'phenotype' values
    phenotype_values = []
    for i in range(len(metadata['points'])):
        phenotype_values.append(metadata['points'][i]['phenotype'])

    unique_drugs = set(phenotype_values)

    name_change = {}

    # replace mitomycinc with 'mitomycin c'
    unique_drugs = [drug.replace('mitomycinc', 'mitomycin c') for drug in unique_drugs]
    name_change['mitomycinc'] = 'mitomycin c'
    # replace cytochalasind with 'cytochalasin d'
    unique_drugs = [drug.replace('cytochalasind', 'cytochalasin d') for drug in unique_drugs]
    name_change['cytochalasind'] = 'cytochalasin d'
    # replace oligomycin with 'oligomycin a'
    unique_drugs = [drug.replace('oligomycin', 'oligomycin a') for drug in unique_drugs]
    name_change['oligomycin'] = 'oligomycin a'
    # replace antimycin with 'antimycin a'
    unique_drugs = [drug.replace('antimycina', 'antimycin a') for drug in unique_drugs]
    name_change['antimycina'] = 'antimycin a'
    # replace latrunculinb with 'latrunculin b'
    unique_drugs = [drug.replace('lantrunculinb', 'latrunculin b') for drug in unique_drugs]
    name_change['lantrunculinb'] = 'latrunculin b'
    # replace dnp with 'dinitrophenol'
    unique_drugs = [drug.replace('dnp', 'dinitrophenol') for drug in unique_drugs]
    name_change['dnp'] = 'dinitrophenol'

    # save the smiles and pubchem link for each drug
    drug_dict = {}
    for drug in unique_drugs:
        try:
            if drug == 'control':
                compound = pcp.get_compounds('DMSO', 'name')[0]
            else:
                compound = pcp.get_compounds(drug, 'name')[0]
            smiles = compound.to_dict(properties=['canonical_smiles'])['canonical_smiles']
            drug_dict[drug] = {
                'smiles': smiles,
                'pubchem': f"{compound.cid}"
            }
        except Exception as e:
            print(f"Error getting compound for {drug}: {e}")
            continue

    print(drug_dict)

    # in the original metadata, add these smiles and pubchem links to the 'treatment' field
    for i in range(len(metadata['points'])):
        drug_name = metadata['points'][i]['treatment']['drug']
        # add the dose to the metadata
        if drug_name in dose_dict:
            metadata['points'][i]['treatment']['dose'] = dose_dict[drug_name]
        else:
            metadata['points'][i]['treatment']['dose'] = 'unknown'
            print(f"Drug {drug_name} not found in dose_dict")
        if drug_name in name_change:
            drug_name = name_change[drug_name]
        if drug_name in drug_dict:
            metadata['points'][i]['treatment']['smiles'] = drug_dict[drug_name]['smiles']
            metadata['points'][i]['treatment']['pubchem'] = drug_dict[drug_name]['pubchem']
        else:
            print(f"Drug {drug_name} not found in drug_dict")

    # save the metadata
    with open(f"{save_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)