import numpy as np
import torch
import torch.nn as nn

n_actions = 2

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def predict_probs(states, model):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :param model: torch model
    :returns: numpy array of shape [batch, n_actions]
    """
    # convert states, compute logits, use softmax to get probability

    # YOUR CODE GOES HERE
    with torch.no_grad():  
        states_tensor = torch.from_numpy(states).float()
        logits = model(states_tensor)
        probs = torch.softmax(logits, dim=1)
        return probs.numpy()

def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    Take a list of immediate rewards r(s,a) for the whole session
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).

    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    A simple way to compute cumulative rewards is to iterate from the last
    to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    # YOUR CODE GOES HERE
    cumulative_rewards = []
    #assert cumulative_rewards is not None, "cumulative_rewards is not defined"
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        cumulative_rewards.append(G)
    cumulative_rewards.reverse()
    return cumulative_rewards  

def get_loss(logits, actions, rewards, n_actions=n_actions, gamma=0.99, entropy_coef=1e-2):
    """
    Compute the loss for the REINFORCE algorithm.
    """
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    probs = torch.softmax(logits, dim=1)
    #assert probs is not None, "probs is not defined"

    log_probs = torch.log_softmax(logits, dim=1)
    #assert log_probs is not None, "log_probs is not defined"

    #assert all(isinstance(v, torch.Tensor) for v in [logits, probs, log_probs]), \
        #"please use compute using torch tensors and don't use predict_probs function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    actions_one_hot = to_one_hot(actions, n_actions)
    log_probs_for_actions = (log_probs * actions_one_hot).sum(dim=1) # [batch,]
    #assert log_probs_for_actions is not None, "log_probs_for_actions is not defined"
    J_hat = (log_probs_for_actions * cumulative_returns).mean()  # a number
    #assert J_hat is not None, "J_hat is not defined"
    
    # Compute loss here. Don't forget entropy regularization with `entropy_coef`
    entropy = -(probs * log_probs).sum(dim=1).mean()
    #assert entropy is not None, "entropy is not defined"
    loss = -J_hat - entropy_coef * entropy
    #assert loss is not None, "loss is not defined"

    return loss


