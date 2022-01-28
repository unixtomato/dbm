import numpy as np

import torch

from datetime import datetime

import os
import sys
import timeit
import shutil
import gzip
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')

device = torch.device("cuda")

class RBM(object):
    def __init__(
        self,
        n_visible=784,
        n_hidden=500,
        W=None,
        vbias=None,
        hbias=None,
        propup_scale=1.0,
        propdown_scale=1.0,
    ):

        np_rng = np.random.RandomState(1234)

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if W is None:
            initial_W = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=np.float32
            )
            # theano shared variables for weights and biases
            W = torch.tensor(initial_W, device=device)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = torch.tensor(
                data=np.zeros(
                    n_hidden,
                    dtype=np.float32
                ),
                device=device
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = torch.tensor(
                data=np.zeros(
                    n_visible,
                    dtype=np.float32
                ),
                device=device
            )

        # initialize input layer for standalone RBM or layer0 of DBN

        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        self.params = [self.W, self.hbias, self.vbias]

        self.propup_scale = propup_scale
        self.propdown_scale = propdown_scale

    def free_energy(self, v_sample):
        wx_b = torch.matmul(v_sample, self.W) + self.hbias
        vbias_term = torch.matmul(v_sample, self.vbias)
        hidden_term = torch.sum(torch.nn.functional.softplus(wx_b), axis=1)
        return -hidden_term - vbias_term

    def energy(self, v, h):
        return torch.sum((torch.matmul(v, self.W))*h, axis=1) + torch.matmul(v, self.vbias) + torch.matmul(h, self.hbias)

    def propup(self, vis):
        pre_sigmoid_activation = self.propup_scale * torch.matmul(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, torch.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = torch.bernoulli(h1_mean)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = self.propdown_scale * torch.matmul(hid, self.W.t()) + self.vbias
        return [pre_sigmoid_activation, torch.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = torch.bernoulli(v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def contrastive(self, input, gibbs_steps=5):
        v1_sample = input
        for k in range(gibbs_steps):
            [pre_sigmoid_h1, h1_mean, h1_sample, 
             pre_sigmoid_v1, v1_mean, v1_sample] = self.gibbs_vhv(v1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, input, optimizer, gibbs_steps=5):

        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(input)

        [
            pre_sigmoid_nh, 
            nh_mean, 
            nh_sample,
            pre_sigmoid_nv, 
            nv_mean, 
            nv_sample
        ] = self.contrastive(input, gibbs_steps)

        #chain_end = nv_sample
        for param in self.params:
            param.requires_grad_(True)
        #cost = torch.mean(self.free_energy(input)) - torch.mean(self.free_energy(chain_end))
        cost = -(torch.mean(self.energy(input,h1_mean)) - torch.mean(self.energy(nv_sample,nh_sample)))
        cost.backward()

        optimizer.step()
        optimizer.zero_grad()
        for param in self.params:
            param.requires_grad_(False)

        monitoring_cost = torch.mean(torch.sum(torch.nn.functional.binary_cross_entropy(nv_mean, input, reduction='none'), axis=1))
        return monitoring_cost

        
class DBM(object):
    def __init__(self, sizes):

        depth = len(sizes)

        weights = []
        biases = []

        np_rng = np.random.RandomState(1234)
        for i in range(depth-1):
            n_visible = sizes[i]
            n_hidden = sizes[i+1]
            initial_W = np.asarray(
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=np.float32
            )
            weights.append(torch.tensor(initial_W, device=device))
            #weights.append(torch.zeros((n_visible, n_hidden), device=device))
        for i in range(depth):
            biases.append(torch.zeros(sizes[i], device=device))


        self.depth = depth
        self.sizes = sizes

        self.weights = weights
        self.biases = biases

        self.log_part_base = np.sum(self.sizes) * np.log(2)
        #self.log_part_base = np.sum([sizes[i*2] for i in range((depth+1)//2)]) * np.log(2)

    def energy(self, variables):
        term = 0
        for i in range(self.depth-1):
            term += torch.sum(torch.matmul(variables[i], self.weights[i]) * variables[i+1], axis=1)
        for i in range(self.depth):
            term += torch.matmul(variables[i], self.biases[i])
        return term

    def propagate(self, i, variables, temperature=1.0):
        term = 0
        if (i != 0):
            term += torch.matmul(variables[i-1], self.weights[i-1])
        if (i != self.depth-1):
            term += torch.matmul(variables[i+1], self.weights[i].t())
        term += self.biases[i]
        term *= temperature
        #term = torch.sigmoid(term)
        return term


    def mean_field(self, variables, steps=30, temperature=1.0):
        #prev_variables = [torch.clone(var) for var in variables]
        #n_sample = variables[0].shape[0]
        #for i in range(1,self.depth):
        #    variables[i] = torch.bernoulli(torch.ones((n_sample, self.sizes[i]), device=device)*0.5)
        for k in range(steps):
            for i in range(1, self.depth):
                variables[i] = torch.sigmoid(self.propagate(i, variables, temperature))
            #print(k, torch.tensor([torch.linalg.norm(v_old - v) for v_old, v in zip(prev_variables, variables)]).mean())

    def gibbs_sampling(self, variables, steps=10, temperature=1.0):
        p_samples = [None] * self.depth
        for k in range(steps):
            for i in range(1, self.depth):
                term = torch.sigmoid(self.propagate(i, variables, temperature))
                p_samples[i] = term
                variables[i] = torch.bernoulli(term)
            term = torch.sigmoid(self.propagate(0, variables, temperature))
            p_samples[0] = term
            variables[0] = torch.bernoulli(term)
        return p_samples

    #def gibbs_sampling(self, variables, steps=10, temperature=1.0):
    #    p_samples = [None] * self.depth
    #    for k in range(steps):
    #        for l in range(self.depth//2):
    #            i = l*2+1
    #            term = torch.sigmoid(self.propagate(i, variables, temperature))
    #            p_samples[i] = term
    #            variables[i] = torch.bernoulli(term)
    #        for l in range((self.depth+1)//2):
    #            i = l*2
    #            term = torch.sigmoid(self.propagate(i, variables, temperature))
    #            p_samples[i] = term
    #            variables[i] = torch.bernoulli(term)
    #    return p_samples

    def log_prob_ratio(self, samples, temp1, temp2):
        # summed over odd layers
        term = 0
        for l in range(self.depth//2):
            i = l*2+1
            term1 = self.propagate(i, samples, temp1)
            term2 = self.propagate(i, samples, temp2)
            term += torch.sum(torch.nn.functional.softplus(term2)-torch.nn.functional.softplus(term1), axis=1)
        return term

    def anneal(self, steps=1000, n_sample=1000):
        samples = []
        log_part_ratio = torch.zeros(n_sample, device=device)
        for i in range(len(self.sizes)):
            samples.append(torch.bernoulli(torch.ones((n_sample, self.sizes[i]), device=device)*0.5))

        # compute log partition function
        temp1 = 0.0
        temp2 = 0.0
        for temp in np.linspace(0,1,steps)[1:]:
            temp2 = temp
            log_part_ratio += self.log_prob_ratio(samples, temp1, temp2)
            self.gibbs_sampling(samples, steps=1, temperature=temp2)
            temp1 = temp

        return torch.mean(log_part_ratio)
    
    def log_prob(self, testset, mf_steps=30):
        # variational lower bound

        batch_size = testset.shape[0]
        posterior = [None]
        for i in range(len(self.sizes)-1):
            posterior.append(torch.bernoulli(torch.ones((batch_size, self.sizes[i+1]), device=device)*0.5))
        posterior[0] = testset
        self.mean_field(posterior, mf_steps)

        res1 = torch.mean(self.energy(posterior))

        hiddens = torch.cat(posterior[1:], dim=1)
        res2 = torch.mean(torch.sum(torch.nn.functional.binary_cross_entropy(hiddens, hiddens, reduction='none'), axis=1))

        log_part_ratio = self.anneal()
        log_part = log_part_ratio + self.log_part_base
        res3 = log_part

        #print(res1, res2, res3)
        res = res1 + res2 - res3

        return res

    def get_cost_updates(self, optimizer, posterior, persistent):
        # note: the first layer in posterior is the clamped visible samples

        self.mean_field(posterior, steps=30)
        p_samples = self.gibbs_sampling(persistent, steps=3)

        for param in self.weights + self.biases:
            param.requires_grad_(True)
        cost = -(torch.mean(self.energy(posterior)) - torch.mean(self.energy(persistent)))
        cost.backward()

        optimizer.step()
        optimizer.zero_grad()
        for param in self.weights+self.biases:
            param.requires_grad_(False)

        monitoring_cost = torch.mean(torch.sum(torch.nn.functional.binary_cross_entropy(persistent[0], posterior[0], reduction='none'), axis=1))
        #monitoring_cost = self.log_prob(posterior)
        return monitoring_cost, posterior, persistent, p_samples


def pre_train_dbm(dbm, trainset):

    learning_rate = 0.001
    gibbs_steps = 15
    training_epochs = 48
    batch_size = 200

    rbms = []
    for i in range(dbm.depth-1):
        propup_scale = 1.0 if i == dbm.depth-2 else 2.0
        propdown_scale = 1.0 if i == 0 else 2.0
        rbm = RBM(
            n_visible=dbm.sizes[i],
            n_hidden=dbm.sizes[i+1],
            W=None,
            vbias=None,
            hbias=None,
            propup_scale=propup_scale,
            propdown_scale=propdown_scale,
        )
        rbms.append(rbm)


    for i in range(dbm.depth-1):
        rbm = rbms[i]
        optimizer = torch.optim.Adam(rbm.params, lr=learning_rate)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=False
        )

        for epoch in range(training_epochs):
            mean_cost = []
            for images in trainloader:
                mean_cost.append(
                    rbm.get_cost_updates(
                        images,
                        optimizer,
                        gibbs_steps
                    ).cpu().numpy()
                )
            print('layer {}, epoch {}, cost is {:.4f}'.format(i, epoch, np.mean(np.mean(mean_cost))))

        # propup dataset
        #rbm.propup_scale = 1.0
        _, trainset = rbm.propup(trainset)

    for i in range(dbm.depth-1):
        dbm.weights[i] = rbms[i].W
    for i in range(dbm.depth):
        if (i == 0):
            dbm.biases[i] = rbms[i].vbias
        elif (i == dbm.depth-1):
            dbm.biases[i] = rbms[i-1].hbias
        else:
            dbm.biases[i] = (rbms[i-1].hbias + rbms[i].vbias) / 2

def persistent_init(dbm, trainset):

    rbms = []
    for i in range(dbm.depth-1):
        propup_scale = 1.0 if i == dbm.depth-2 else 2.0
        propdown_scale = 1.0 if i == 0 else 2.0
        rbm = RBM(
            n_visible=dbm.sizes[i],
            n_hidden=dbm.sizes[i+1],
            W=None,
            vbias=None,
            hbias=None,
            propup_scale=propup_scale,
            propdown_scale=propdown_scale,
        )
        rbms.append(rbm)


    persistent = []
    persistent.append(trainset)
    for i in range(dbm.depth-1):
        _, trainset = rbms[i].propup(trainset)
        persistent.append(trainset)

    return persistent

def test_rbm(
    learning_rate=0.001, 
    training_epochs=20, 
    batch_size=20, 
    output_folder='plots'
):

    trainset = torch.from_numpy(np.asarray(np.load('images.npy'), dtype=np.float32).reshape(-1, 784))
    trainset = trainset.to(device)

    #with gzip.open("mnist.pkl.gz", 'rb') as f:
    #    trainset, validset, testset = pickle.load(f, encoding='latin1')
    #trainset = torch.from_numpy(np.asarray(trainset[0], dtype=np.float32).reshape(-1, 784))
    #trainset = trainset.to(device)

    print("# of trainset", trainset.shape[0])

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)


    start_time = timeit.default_timer()

    sizes = [784, 512, 1024]
    dbm = DBM(sizes)

    #print('simple layer-rise pretrain')
    #pre_train_dbm(dbm, trainset)
    #print('log-prob {}'.format(dbm.log_prob(trainset).cpu().numpy()))
    #sys.exit()

    posterior = [None]
    for i in range(len(sizes)-1):
        posterior.append(torch.bernoulli(torch.ones((batch_size, sizes[i+1]), device=device)*0.5))


    #persistent = []
    #for i in range(len(sizes)):
    #    persistent.append(torch.bernoulli(torch.ones((batch_size, sizes[i]), device=device)*0.5))
    persistent = persistent_init(dbm, trainset[:batch_size])


    optimizer = torch.optim.Adam(dbm.weights + dbm.biases, lr=learning_rate)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False
    )

    for epoch in range(training_epochs):
        mean_cost = []
        for data in trainloader:
            posterior[0] = data
            cost, posterior, persistent, p_samples = dbm.get_cost_updates(optimizer, posterior, persistent)
            mean_cost.append(cost.cpu().numpy())
        #print('epoch {}, cost is {:.4f}'.format(epoch, np.mean(np.mean(mean_cost))))
        print('epoch {}, log-prob {}'.format(epoch, dbm.log_prob(trainset).cpu().numpy()))
        

    # generate samples

    wid = 10
    persistent = []
    for i in range(len(sizes)):
        persistent.append(torch.bernoulli(torch.ones((wid*wid, sizes[i]), device=device)*0.5))
    p_samples = dbm.gibbs_sampling(persistent, 15)

    chain_end = p_samples[0].cpu().numpy()
    tile = np.empty((wid*28,wid*28))
    for i in range(wid):
        for j in range(wid):
            tile[i*28:(i+1)*28,j*28:(j+1)*28] = chain_end[i*wid+j].reshape(28,28)
    plt.imshow(tile, 'gray')
    plt.axis('off')
    #plt.colorbar()
    plt.savefig(output_folder + '/plotd04.png')
    plt.clf()


    print('time elapsed {}'.format(timeit.default_timer() - start_time))


if __name__ == '__main__':
    test_rbm()


