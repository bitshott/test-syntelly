from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

import deepchem as dc

from config.experiment_config import config


class FeatureExtractor:
    def __init__(self):
        try:
            self.mol2vec_model = dc.feat.Mol2VecFingerprint(str(config.MODEL_PATH / 'model_300dim.pkl'))
        except ImportError:
            pass
        self.melt_columns_to_drop = ['fr_Ar_NH',
                                    'NumAmideBonds',
                                    'MaxAbsEStateIndex',
                                    'MolWt',
                                    'NumAromaticCarbocycles',
                                    'fr_COO',
                                    'Chi1',
                                    'MolWt',
                                    'HeavyAtomMolWt',
                                    'NumValenceElectrons',
                                    'Chi0',
                                    'fr_phenol',
                                    'fr_phos_acid',
                                    'NumValenceElectrons',
                                    'Kappa2',
                                    'LabuteASA',
                                    'Chi1',
                                    'Chi0n',
                                    'Chi0',
                                    'NumValenceElectrons',
                                    'NumValenceElectrons',
                                    'SMR_VSA2',
                                    'Chi0',
                                    'NumValenceElectrons',
                                    'Chi3n',
                                    'Chi0v',
                                    'Chi0v',
                                    'LabuteASA',
                                    'Chi0n',
                                    'Chi0n',
                                    'Kappa2',
                                    'NumValenceElectrons',
                                    'Chi0n',
                                    'Chi1n',
                                    'Chi0n',
                                    'NumRotatableBonds',
                                    'Chi0',
                                    'Chi0n',
                                    'Chi0n',
                                    'SMR_VSA7',
                                    'fr_Al_OH',
                                    'Chi1n',
                                    'NumValenceElectrons',
                                    'Chi0',
                                    'MaxPartialCharge',
                                    'MinPartialCharge',
                                    'MaxAbsPartialCharge',
                                    'MinAbsPartialCharge',
                                    'Ipc',
                                    'BCUT2D_MWHI',
                                    'BCUT2D_MWLOW',
                                    'BCUT2D_CHGHI',
                                    'BCUT2D_CHGLO',
                                    'BCUT2D_LOGPHI',
                                    'BCUT2D_LOGPLOW',
                                    'BCUT2D_MRHI',
                                    'BCUT2D_MRLOW']

    def smiles_to_rdkit_desc(self, smiles: str, columns_to_drop: list = None) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(smiles)
            return None
    
        if columns_to_drop is None:
            return {nm: fn(mol) for nm, fn in Descriptors._descList}
        else:
            return {nm: fn(mol) for nm, fn in Descriptors._descList if nm not in columns_to_drop}


    def smiles_to_mol2vec_embeddings(self, smiles: str, start_index: int = 0):
        embeddings = self.mol2vec_model.featurize(smiles)
        return {idx: value for idx, value in enumerate(embeddings[0], start=start_index)}
    
    def smiles_to_morgan_fp(self, smiles: str, start_index: int = 0) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        vec = generator.GetFingerprint(mol)
        fp_feat = {idx: int(bit) for idx, bit in enumerate(vec)}
        return fp_feat
        
    def smiles_to_rdkit_and_morgan(self, smiles: str, columns_to_drop: list = None) -> dict:
        desc = self.smiles_to_rdkit_desc(smiles, columns_to_drop=columns_to_drop)
        fp = self.smiles_to_morgan_fp(smiles)
        desc.update(fp)
        return desc
    
featurizer = FeatureExtractor()