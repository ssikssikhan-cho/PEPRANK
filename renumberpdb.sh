#!/usr/bin/env sh

usage ()
{
    echo "Usage"
    echo "$0 'selection' pdbfile.pdb"
    echo "Selection string can be:"
    echo "    • residues: renumber the resids of the pdb"
    echo "    • atoms: renumber the atom numbers"
    exit
}

if [ $# -ne 2 ]; then
    usage
fi

SELECTION=$1
PDB=$2

awk -v SELECTION="$SELECTION" '
    function printpdb(selection){
        if (selection=="atoms"){
            atomid+=1
            printf("%-6s%5s%1s%4s%1s%3s%1s%1s%4s%1s%3s%8s%8s%8s%6s%6s%6s%4s\n", $1,atomid,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
        }
        if (selection=="residues"){
            if ($9 != resid_prev){
                resid+=1
                resid_prev = $9
            }
            printf("%-6s%5s%1s%4s%1s%3s%1s%1s%4s%1s%3s%8s%8s%8s%6s%6s%6s%4s\n", $1,$2,$3,$4,$5,$6,$7,$8,resid,$10,$11,$12,$13,$14,$15,$16,$17,$18)
        }
    }
    BEGIN{
        FIELDWIDTHS = "6 5 1 4 1 3 1 1 4 1 3 8 8 8 6 6 6 4"
        # $2: Atom serial number
        # $4: Atom type
        # $5: altLoc; alternate location indicator.
        # $6: Resname
        # $8: ChainID
        # $9: Resid
        # $12: x
        # $13: y
        # $14: z
        atomid=0
        resid=0
        resid_prev=0
    }
!/^USER/||/^REMARK/||/^END/ {
    printpdb(SELECTION)
}' "$PDB"
