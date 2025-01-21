# Copyright (c) 2024 Joon-Sang Park

from Bio.PDB import *

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def get_atom_list(structure, whichchain):
	output = dict()
	for chain in structure:
		if chain.id == whichchain.id:
			for residue in chain.get_residues():
				hetflag, resseq, icode=residue.get_id()
				the_id = (chain.id+"_"+str(resseq)+"_"+icode).strip()
				for atom in residue.get_unpacked_list():
					if hetflag==' ':
						if the_id in output:
							output[the_id].append(atom)
						else:
							output[the_id] = [atom]
	return output

def is_contact(res_1,other_atoms,cutoff):
	for atom in res_1:
		ns = NeighborSearch(other_atoms)
		center = atom.get_coord()
		neighbors = ns.search(center, cutoff) 
		residue_list = Selection.unfold_entities(neighbors, 'R') # R for residues
		if len(residue_list)>0:
			return True
	return False 	

def get_all_atoms(residue_map):
	all_atoms_out = []
	for residue in residue_map:
		for atom in residue_map[residue]:
			all_atoms_out.append(atom)
	return all_atoms_out

def get_contact_list(struc,all_atoms,cutoff):
	contacts = []
	for residue in struc:
		atom_list = struc[residue]
		outcome = is_contact(atom_list,all_atoms,cutoff)
		if outcome:
			contacts.append(residue)
	return contacts				

def save_if(filename_in,filename_out,contactlist):
	f = open(filename_in,'r')
	f_out = open(filename_out,'w')
	for line in f.readlines():
		if "ATOM" in line:
			try:
				line = line.strip()
				resid=line[22:28].strip()
				icode = "" 
				if (not is_number(resid[-1])):
					icode = resid[-1]
					resid = resid[:-1]
				resname = line[21]+"_"+resid+"_"+icode
				if resname in contactlist:
					f_out.write(line+"\n")
			except:
				continue

	
def extract_interface(filename, cutoff = 10.0):
	#must catch exceptions outside

	complexstruct = PDBParser(QUIET=True).get_structure('complex', filename)
	#assume there are only two chains
	chains = complexstruct.get_chains()
	rec_chain = next(chains)
	lig_chain = next(chains)
	#for chain in chains:
	#	raise Exception('There are more than two chains!')

	complexstruct = PDBParser(QUIET=True).get_structure('complex', filename)

	entity = Selection.unfold_entities(complexstruct, 'C')
	rec_al = get_atom_list(entity,rec_chain)
	if len(rec_al) == 0:
		raise Exception(f'no atoms in chain {rec_chain}')
	lig_al = get_atom_list(entity,lig_chain)
	if len(lig_al) == 0:
		raise Exception(f'no atoms in chain {lig_chain}')

	rec_all_atoms = get_all_atoms(rec_al)
	lig_all_atoms = get_all_atoms(lig_al)
	#The chain with more residue is the receptor
	#print( len(rec_all_atoms) ,  len(lig_all_atoms))
	if len(rec_all_atoms) < len(lig_all_atoms):
		tmp = rec_all_atoms, rec_al
		rec_all_atoms, rec_al = lig_all_atoms, lig_al
		lig_all_atoms, lig_al = tmp

	rec_contacts = get_contact_list(rec_al, lig_all_atoms, cutoff)
	lig_contacts = get_contact_list(lig_al, rec_all_atoms, cutoff)

	rif = filename[:-3]+"rec-if.pdb"
	lif = filename[:-3]+"lig-if.pdb"
	save_if(filename,rif,rec_contacts)
	save_if(filename,lif,lig_contacts)

	return (rif, lif) 
