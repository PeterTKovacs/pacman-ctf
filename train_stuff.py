
import random, time, util
import numpy as np
import torch
import torch.nn as nn
import copy

class dqn(nn.Module):

  def __init__(self,n_in,replay_buffer,lrate=5e-4):

    super(dqn,self).__init__()

    

    self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_in, 100),
            nn.ReLU(),
            nn.Linear(100, 70),
            nn.ReLU(),
            nn.Linear(70, 40),
            nn.ReLU(),
            nn.Linear(40,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )

    self.replay_buffer=replay_buffer

    self.optimizer=torch.optim.SGD(self.parameters(),lrate)

  def forward(self,x):

    return self.linear_relu_stack(x)

  def predict_value(self,x,detach=True):

    self.eval()
    if detach:
      return self.forward(self.predict_transform(x)).detach().cpu().numpy()[0]
    else:
      return self.forward(self.predict_transform(x)).numpy()[0]

  def trainstep(self,batch,device='cuda'): 
        
        self.train()
        
        try:
            self.optimizer.zero_grad()
        except:
            print('faliled to set optimizer')
            return -1
        
        x,y=self.batch_transform(batch,device) # batch transform eg. to get rid of irrelevant labels
        y_=self.forward(x)
        
        loss=self.loss(y_,y)
        
        loss.backward()
        self.optimizer.step()

  def loss(self,y_,y):
    return ((y-y_)**2).sum()

  def generate_batch(self,replay_buffer,indices):

    x=[replay_buffer[i][0] for i in indices]
    y=[replay_buffer[i][1] for i in indices]

    return (torch.Tensor(x).to(torch.float),torch.Tensor(y).to(torch.float))

  def batch_transform(self,batch,device):
    return batch[0].to(device),batch[1].to(device)

  def predict_transform(self,x,device="cuda"):
    xx=torch.from_numpy(x).to(torch.float)
    torch.reshape(xx,(1,-1))

    return xx.to(device)


class train_manager:
  '''
  class that will organize the training of the nets in a DQN manner
  this is the interface (on behalf of the engine) the agent objects communicate with
  '''

  def __init__(self,team_indices,n_replay,batch_size,update_frequency,device,model,model_args={}):

    self.n_replay=n_replay
    self.team_indices=team_indices
    self.batch_size=batch_size
    self.update_frequency=update_frequency
    self.device=device

    self.past_states={i:None for i in team_indices} # store the features
    self.past_features={i:None for i in team_indices}
    self.rewards={i:None for i in team_indices}
    self.current_states={i:None for i in team_indices}
    self.current_features={i:None for i in team_indices}

    self.frozen_model=model(**model_args).to(device)
    self.warm_model=model(**model_args).to(device)

    self.replay_buffer=[]
   

    self.buffer_write_idx=0
    self.backprop_counter=0

  def pick_action(self,admissable_actions,possible_afterstates,possible_rewards):

    '''
    upon receiving the admissable set with !embedded! afterstates from the agent, evaluate those taking possible rewards into account
    (heuristic [pessiistic] rollout needs to be implemented by agent)
    '''

    state_values=[self.warm_model.predict_value(state) for state in possible_afterstates]

    optimal_action=admissable_actions[0]
    opt_value=possible_rewards[0]+state_values[0]

    for i in range(1,len(admissable_actions)):
      if (possible_rewards[i]+state_values[i])>opt_value:
        optimal_action=admissable_actions[i]
        opt_value=possible_rewards[i]+state_values[i]

    return optimal_action

  def log_state(self,state,feature,idx):
    self.current_states[idx]=state.deepCopy()
    self.current_features[idx]=feature
  
  def administer_state_transition(self,idx):

    if self.past_states[idx]!=None:

      reward=self.reward_generator(self.past_states[idx],self.current_states[idx])

      y=self.frozen_model.predict_value(self.current_features[idx])+reward

      if len(self.replay_buffer)<self.n_replay:
        self.replay_buffer.append((self.past_features[idx],y)) # input features and what should have been predicted (bootstrap sense)

      else:
        self.replay_buffer[self.buffer_write_idx]=(self.past_features[idx],y)

      self.buffer_write_idx=(self.buffer_write_idx+1)%self.n_replay

    self.past_states[idx]=self.current_states[idx]
    self.past_features[idx]=self.current_features[idx]

  def reward_generator(self,old_state,new_state):

    return -1

  def training_step(self):

   
    

    indices=list(random.choices(range(len(self.replay_buffer)),k=self.batch_size))
    batch=self.warm_model.generate_batch(self.replay_buffer,indices)

    self.warm_model.trainstep(batch,self.device)

    print("trainstep: ",self.backprop_counter)

    self.backprop_counter+=1

    if self.backprop_counter==self.update_frequency:
    
      self.frozen_model=copy.deepcopy(self.warm_model)
      self.frozen_model.eval()
      self.backprop_counter=0

  def save_warm_model(self,path):
    torch.save(self.warm_model.state_dict(), path)

  def load_model(self,path,warm=True):
    
    if warm:
      self.warm_model.load_state_dict(torch.load(path))
      self.train()
    else:
      self.frozen_model.load_state_dict(torch.load(path))
      self.eval()

  def reset_state_logs(self):
    self.past_states={i:None for i in self.team_indices} # store the features
    self.past_features={i:None for i in self.team_indices}
    self.rewards={i:None for i in self.team_indices}
    self.current_states={i:None for i in self.team_indices}
    self.current_features={i:None for i in self.team_indices}

      
  
