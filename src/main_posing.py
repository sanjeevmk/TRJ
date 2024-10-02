import torch
torch.manual_seed(4)
torch.cuda.manual_seed_all(4)
import sys
from init import initialize_args,init_optimizers,init_njf_wks_posing_network
from init import init_amass_sequence_data,init_posing_network,init_pointnet_network,init_transformer_network,init_joints_encoder_network,init_d3f_pointnet_network
#from spacetime import Method
import logging
import os
from posing_d3f import D3FPosingTrainer

def run(data_args,training_args,loss_args,test_args,log_args):
   dataset = init_amass_sequence_data(data_args.root,data_args.bodymodel,data_args.seq_length,data_args.max_frames)

   if test_args.test:
      test_dataset = init_amass_sequence_data(test_args.testdir,data_args.bodymodel,data_args.seq_length,data_args.max_frames) 

   posing_network = init_njf_wks_posing_network()
   #posing_network.load_state_dict(torch.load(training_args.pose_weights))
   pointnet_network = init_d3f_pointnet_network()
   optimizer = init_optimizers([posing_network,pointnet_network],training_args.lr)

   if not test_args.test:
         method = D3FPosingTrainer(posing_network,pointnet_network,dataset,training_args.batch,data_args.seq_length,log_args.dir)
         method.loop_epochs(optimizer,training_args.epochs,loss_args,training_args,test_args,data_args)
   else:
         method = D3FPosingTrainer(posing_network,pointnet_network,dataset,training_args.batch,data_args.seq_length,log_args.dir,test_dataset=test_dataset)
         if test_args.mode == "nonhuman":
            method.def_trans_nonhuman(optimizer,training_args.epochs,loss_args,training_args,test_args,data_args)
         else:
            method.def_trans_multiple(optimizer,training_args.epochs,loss_args,training_args,test_args,data_args)

if __name__ == '__main__':
   config_file = sys.argv[1]
   data_args,training_args,loss_args,test_args,log_args = initialize_args(config_file)
   open(os.path.join(log_args.dir,"logs"),'w')
   logging.basicConfig(filename=os.path.join(log_args.dir,"logs"),
    filemode='a',
    format='%(asctime)s,%(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
   run(data_args,training_args,loss_args,test_args,log_args)