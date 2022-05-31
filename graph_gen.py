
from datagen.graphgen import gen_erdos_renyi, gen_barabasi_albert, gen_twod_grid
from datagen.algorithms import info, gen_multi_algo_data
import sys
import torch
import os

#get_streams.py & graphgen.py ('_' after save_fp) need minor modification


class Gen_graph():
	def __init__(self, ngraph_train, ngraph_val, ngraph_test, nnode, nnode_test, algo_names):
		self.ngraph_train = ngraph_train
		self.ngraph_val = ngraph_val
		self.ngraph_test = ngraph_test
		self.nnode = nnode
		self.nnode_test = nnode_test
		self.algo_names = algo_names
          
	def gen_ErdosRenyi(self, rand_generator, num_graph, num_node, task_list, datasavefp, directed=False):
		gen_erdos_renyi(rand_generator, int(num_graph), int(num_node), datasavefp+"_", directed)
		src_nodes = torch.argmax(rand_generator.sample((int(num_graph), int(num_node))), dim=1)
		gen_multi_algo_data(
			datasavefp+"_"+"_erdosrenyi" + num_graph + "_"+ num_node + ".pt",
			src_nodes,
			task_list,
			True
		)
 
	def gen_BarabasiAlbert(self,rand_gen, num_graph, num_node, task_list, datasavefp):
		m = 0
		gen_barabasi_albert(rand_gen, int(num_graph), int(num_node), m, datasavefp)
		src_nodes = torch.argmax(rand_gen.sample((int(num_graph), int(num_node))), dim=1)
		gen_multi_algo_data(
			datasavefp+"_"+"_barabasialbert" + num_graph + "_"+ num_node + ".pt",
			src_nodes,
			task_list,
			True
		)	


	def gen_twoGrid(self,rand_gen, num_graph, num_node, task_list, datasavefp):
		gen_twod_grid(rand_gen, int(num_graph), int(num_node), datasavefp)
		src_nodes = torch.argmax(rand_gen.sample((int(num_graph), int(num_node))), dim=1)
		gen_multi_algo_data(
			datasavefp+"_"+"_twodgrid" + num_graph + "_"+ num_node + ".pt",
			src_nodes,
			task_list,
			True
		)
	def generate(self, graphtype, train=False):
		rand_gen = torch.distributions.Uniform(0.0, 1.0)
		if graphtype == 'erdosrenyi':       
			if not os.path.exists('Data/erdosrenyi'):
				os.makedirs('Data/erdosrenyi')
			for i in range(len(self.ngraph_train)): 
				self.gen_ErdosRenyi(rand_gen, self.ngraph_train[i], self.nnode, self.algo_names, 'Data/erdosrenyi/train')
			self.gen_ErdosRenyi(rand_gen, self.ngraph_val, self.nnode, self.algo_names, 'Data/erdosrenyi/val')
			for i in range(len(self.nnode_test)):
				self.gen_ErdosRenyi(rand_gen, self.ngraph_test[i], self.nnode_test[i], self.algo_names, 'Data/ErdosRenyi/test') 
		
		elif graphtype == 'barabasialbert':
			if not os.path.exists('Data/barabasialbert'):
				os.makedirs('Data/barabasialbert')
			for i in range(len(self.ngraph_train)): 
				self.gen_BarabasiAlbert(rand_gen, self.ngraph_train[i], self.nnode, self.algo_names, 'Data/barabasialbert/train')
			self.gen_BarabasiAlbert(rand_gen, self.ngraph_val, self.nnode, self.algo_names, 'Data/barabasialbert/val')
			for i in range(len(self.nnode_test)):
				self.gen_BarabasiAlbert(rand_gen, self.ngraph_test[i], self.nnode_test[i], self.algo_names, 'Data/barabasialbert/test') 


		elif graphtype == 'twodgrid':
			if not os.path.exists('Data/twodgrid'):
				os.makedirs('Data/twodgrid')
			for i in range(len(self.ngraph_train)): 
				self.gen_twoGrid(rand_gen, self.ngraph_train[i], self.nnode, self.algo_names, 'Data/twodgrid/train')
			self.gen_twoGrid(rand_gen, self.ngraph_val, self.nnode, self.algo_names, 'Data/twodgrid/val')
			for i in range(len(self.nnode_test)):
				self.gen_twoGrid(rand_gen, self.ngraph_test[i], self.nnode_test[i], self.algo_names, 'Data/twodgrid/test') 


		if not train:
			sys.exit('Graph Generation Done!')
