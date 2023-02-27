import random
import gym
import pylab
import numpy as np
import torch
import collections
from tqdm import tqdm,trange
from torch.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform
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


	def forward(self,X):
		assert X.shape[0]==self.inp_size
		op=self.model(X) # 8,
		# print(op)
		mean=op[:4]
		cov=torch.diag(torch.exp(op[4:]))
		mvn = torch.distributions.multivariate_normal.MultivariateNormal(mean,cov)
		transformed_dist = torch.distributions.TransformedDistribution(mvn, ComposeTransform([self.sigmoid_transform, self.affine_transform]))
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
		self.train_eps=1000
		self.gamma=0.99
		self.critic_loss_func=torch.nn.MSELoss()

	def train_in_env(self,render=False):
		actor_optimizer=torch.optim.Adam(self.actor.parameters())
		critic_optimizer=torch.optim.Adam(self.critic.parameters())
		last10steps=[0]*10
		last10score=[0]*10
		pbar=trange(self.train_eps)
		for ep_num in pbar:
			state,_ = self.env.reset()
			if ep_num%100==0:
				_,_ = self.val_env.reset()
			log_probs,state_values,rewards = [],[],[]
			done=False
			for _ in range(self.max_episode_steps):
				state=torch.Tensor(state)
				try:
					action_distri=self.actor(state) # torch.distribution
				except:
					import pdb
					pdb.set_trace()
				value=self.critic(state)
				sampled_action=action_distri.sample()*(1+torch.randn(4)*0.05) #if np.random.rand() < 0.8 else (torch.rand(4)*2 - 1)
				sampled_action=sampled_action.clamp(-0.9999999,0.9999999)
				next_state, reward, done, _,_ = self.env.step(sampled_action.cpu().numpy())
				if ep_num%100==0:
					self.val_env.step(sampled_action.cpu().numpy())
				
				
				log_probs.append(action_distri.log_prob(sampled_action)) # maybe add tnah correction
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

			with torch.no_grad():
				return_vals=[rewards[i]+self.gamma*state_values[i+1].item() for i in range(len(state_values)-1)]+[rewards[-1]+self.gamma*terminal_R]
			return_vals=torch.tensor(return_vals).type(torch.float32).to(DEVICE)
			log_probs=torch.stack(log_probs).to(DEVICE)
			state_values=torch.cat(state_values).to(DEVICE)

			adv=return_vals-state_values
			actor_loss=-torch.mean(log_probs*(adv.detach())).type(torch.float32)

			critic_loss=self.critic_loss_func(state_values,return_vals).type(torch.float32)
			actor_optimizer.zero_grad()
			critic_optimizer.zero_grad()
			actor_loss.backward()
			critic_loss.backward()
			actor_optimizer.step()
			critic_optimizer.step()
		torch.save(self.actor, 'actor.pkl')
		torch.save(self.critic, 'critic.pkl')
		self.env.close()
		self.val_env.close()





if __name__=='__main__':

	a=Environment()
	a.train_in_env(True)
