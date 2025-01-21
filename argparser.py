import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F',type=str,help='path to inference/training target(s)') #Path for decoy dir
    parser.add_argument('-F2',type=str,help='path to inference/training target(s)') #Path for second decoy dir
    parser.add_argument('--gpu',type=str,default='0',help='Choose gpu id, example: \'1,2\'(specify use gpu 1 and 2)')
    parser.add_argument("--batch_size", help="batch_size", type=int, default=16)
    parser.add_argument("--num_workers", help="number of (torch) workers", type=int, default=16)
    parser.add_argument("--n_heads", help="number of attention heads", type=int, default=3)
    parser.add_argument("--n_gat_layers", help="number of GAT layers", type=int, default=4)
    parser.add_argument("--dim_gat_feat", help="GAT feature dim", type=int, default=70)
    parser.add_argument("--n_fcl", help="number of FC layers", type=int, default=4)
    parser.add_argument("--dim_fcl_feat", help="dimension of FC layer", type=int, default=64)
    parser.add_argument("--dropout", help="dropout rate", type=float, default=0.3)
    parser.add_argument('--parampath',help='model param file path for infernce',type=str,default=None)
    parser.add_argument('--maxfnum', help='Max # of files used for eval/train', type=int, default = 100000)
    parser.add_argument('--n_epochs', help='', type=int, default = 40)
    args = parser.parse_args()
    params = vars(args)
    return params
