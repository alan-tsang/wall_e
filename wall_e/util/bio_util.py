import warnings
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw


def generate_mol_img(mol, save = '', size = (400, 300)):
    # SMILES
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    elif isinstance(mol, rdkit.Chem.rdchem.Mol):
        pass
    else:
        warnings.warn("mol should be rdkit.Chem.rdchem.Mol or SMILES")

    img = Draw.MolToImage(mol, size=size)
    if save != '':
        img.save(save)

    return img
