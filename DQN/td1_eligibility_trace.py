import random
import gym
import pylab
import numpy as np
import torch
import collections
from tqdm import tqdm,trange
from torch.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform,TanhTransform
import matplotlib.pyplot as plt

torch.manual_seed(10)
np.random.seed(10)
DEVICE='cpu'
torch.set_default_dtype(torch.float32)
class Actor(torch.nn.Module):
	def __init__(self,inp_size):
		super(Actor, self).__init__()
		self.inp_size=inp_size
		layers = [
			torch.nn.Linear(inp_size,64),
			torch.nn.ReLU(),
			torch.nn.Linear(64,128),
			torch.nn.ReLU(),
			torch.nn.Linear(128,8)
		]

		# Initialize the layers with Xavier initialization
		for i in range(len(layers)):
		    if isinstance(layers[i], torch.nn.Linear):
		        torch.nn.init.xavier_uniform_(layers[i].weight)
		        torch.nn.init.constant_(layers[i].bias, 0.0)

		
		# Create a sequential model with the initialized layers
		self.model = torch.nn.Sequential(*layers)
		
		# self.l1=torch.nn.Linear(inp_size,64)
		# torch.nn.init.xavier_uniform_(self.model.weight)
		# self.l2=torch.nn.Linear(64,8)
		# torch.nn.init.xavier_uniform_(self.l2.weight)
		self.relu=torch.nn.functional.relu
		self.sigmoid_transform = SigmoidTransform()
		self.affine_transform = AffineTransform(scale=2, loc=-1)
		self.tanh_transform=TanhTransform()


	def forward(self,X):
		assert X.shape[0]==self.inp_size
		op=self.model(X) # 8,
		# print(op)
		mean=op[:4]
		cov=torch.diag(torch.exp(op[4:]))
		mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean,cov)
		# transformed_dist = torch.distributions.TransformedDistribution(mvn, ComposeTransform([self.sigmoid_transform, self.affine_transform]))
		transformed_dist=torch.distributions.TransformedDistribution(mvn,self.tanh_transform)
		return transformed_dist
		# return multivar_normal_dist.transform(self.sigmoid_transform).transform(self.affine_transform)


class Critic(torch.nn.Module):
	def __init__(self,inp_size):
		super(Critic, self).__init__()
		self.inp_size=inp_size
		layers = [
			torch.nn.Linear(inp_size,64),
			torch.nn.ReLU(),
			torch.nn.Linear(64,128),
			torch.nn.ReLU(),
			torch.nn.Linear(128,1)
		]

		# Initialize the layers with Xavier initialization
		for i in range(len(layers)):
		    if isinstance(layers[i], torch.nn.Linear):
		        torch.nn.init.xavier_uniform_(layers[i].weight)
		        torch.nn.init.constant_(layers[i].bias, 0.0)

		
		# Create a sequential model with the initialized layers
		self.model = torch.nn.Sequential(*layers)

	def forward(self,X):
		assert X.shape[0]==self.inp_size
		op=self.model(X) # 1
		return op


class Environment():
	def __init__(self):
		self.max_episode_steps=1000
		self.env = gym.make('BipedalWalker-v3')
		self.val_env=gym.make('BipedalWalker-v3',render_mode='human')
		# print(self.env.action_space.shape[0])
		# print(self.env.action_space)
		# print("State space: {}".format(env.observation_space))
		# print("Action space: {}".format(env.action_space))
		self.env.action_space.seed(10)
		self.env.observation_space.seed(10)
		self.actor=Actor(24)
		self.critic=Critic(24)
		self.train_eps=10000
		self.gamma=0.99
		self.critic_loss_func=torch.nn.MSELoss()
		self.actor_trace_decay=0.9
		self.critic_trace_decay=0.9

	def train_in_env(self,render=False):
		actor_optimizer=torch.optim.Adam(self.actor.parameters())
		critic_optimizer=torch.optim.Adam(self.critic.parameters())
		last10steps=[0]*10
		last10score=[0]*10
		pbar=trange(self.train_eps)
		
		for ep_num in pbar:
			state,_ = self.env.reset()
			if (ep_num+1)%100==0:
				_,_ = self.val_env.reset()
			log_probs,state_values,rewards = [],[],[]
			done=False
			
			I=1
			actor_traces=[]
			critic_traces=[]
			curr_actor_trace=0
			curr_critic_trace=0
			for _ in range(self.max_episode_steps):
				state=torch.Tensor(state)
				try:
					action_distri=self.actor(state) # torch.distribution
				except:
					import pdb
					pdb.set_trace()
				value=self.critic(state)
				sampled_action=action_distri.sample()*(1+torch.randn(4)*0.05) #if np.random.rand() < 0.8 else (torch.rand(4)*2 - 1)
				sampled_action=sampled_action.clamp(-0.999,0.999)
				next_state, reward, done, _,_ = self.env.step(sampled_action.cpu().numpy())
				if (ep_num+1)%100==0:
					self.val_env.step(sampled_action.cpu().numpy())
				
				
				curr_actor_trace=curr_actor_trace*self.actor_trace_decay*self.gamma+I*action_distri.log_prob(sampled_action)
				actor_traces.append(curr_actor_trace)
				curr_critic_trace=curr_critic_trace*self.critic_trace_decay*self.gamma+I*value
				critic_traces.append(curr_critic_trace)
				
				I=I*self.gamma
				state_values.append(value)
				rewards.append(reward)
				state=next_state
				if done:
					break

			
			if done:
				terminal_R=0
			else:
				with torch.no_grad():
					terminal_R= self.critic(torch.tensor(state))

			last10steps[ep_num%10]=len(rewards) + 1
			last10score[ep_num%10]=sum(rewards) + terminal_R if isinstance(terminal_R, int) else terminal_R.item()
			pbar.set_description(f'avg steps,score:{np.mean(last10steps)},{np.mean(last10score)}')


			######ONE WAY to find return vals
			# with torch.no_grad():
			# 	return_vals=[rewards[i]+self.gamma*state_values[i+1].item() for i in range(len(state_values)-1)]+[rewards[-1]+self.gamma*terminal_R]
			# return_vals=torch.tensor(return_vals).type(torch.float32).to(DEVICE)
			###############################

			######METHOD 2 to find return vals
			# return_vals=[terminal_R]
			# for r in rewards[::-1]:
			# 	return_vals.append(r+self.gamma*return_vals[-1])
			# return_vals= (return_vals[1:])[::-1] # to remove terminal R
			# return_vals=torch.tensor(return_vals).type(torch.float32).to(DEVICE)
			#################################

			#######FAST return vals
			rewards.append(terminal_R)
			rewards=torch.tensor(rewards).type(torch.float32)
			pos_gam_array=torch.pow(self.gamma,torch.arange(rewards.shape[0]))
			neg_gam_arr=1/pos_gam_array
			gmat=neg_gam_arr.unsqueeze(1)@pos_gam_array.unsqueeze(0) # gmat[i][j]=gamma^(j-i)
			gmat=torch.triu(gmat)
			return_vals=(gmat@rewards)[:-1]
			return_vals=return_vals.to(DEVICE)
			###############################

			

			state_values=torch.cat(state_values).to(DEVICE)
			actor_traces=torch.stack(actor_traces,dim=0).to(DEVICE)
			critic_traces=torch.cat(critic_traces,dim=0).to(DEVICE)
			
			adv=return_vals-state_values
			
			######### MEAN or SUM, find what does better########
			actor_loss=-torch.sum(actor_traces*(adv.detach()))		
			critic_loss=-torch.sum(critic_traces*(adv.detach()))
			
			actor_optimizer.zero_grad()
			critic_optimizer.zero_grad()
			actor_loss.backward()
			critic_loss.backward()
			actor_optimizer.step()
			critic_optimizer.step()
			if ep_num % 1000 == 0:
				torch.save(self.actor.state_dict(), f'saved_models/td1 runs with sum in loss/actor{ep_num}.pkl')
				torch.save(self.critic.state_dict(), f'saved_models/td1 runs with sum in loss/critic{ep_num}.pkl')
		self.env.close()
		self.val_env.close()





if __name__=='__main__':

	a=Environment()
	a.train_in_env(True)
