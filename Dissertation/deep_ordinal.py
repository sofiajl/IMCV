import torch
import torch.nn.functional as F

################################################################################
# Small utilities.                                                             #
################################################################################

def fact(x):
    return torch.exp(torch.lgamma(x+1))
def log_fact(x):
    return torch.lgamma(x+1)

################################################################################
# Base class.                                                                  #
################################################################################

class OrdinalMethod(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

    ## BUILD

    def how_many_outputs(self):
        # how many output neurons does this loss require?
        # typically noutputs=nclasses, but not always.
        return self.K

    ## TRAIN

    def post_epoch(self):
        # some losses may use this method to reset running statistics, e.g., to
        # compute thresholds
        pass

    def compute_loss(self, ypred, ytrue):
        # computes the loss
        pass

    ## INFERENCE

    def to_probabilities(self, ypred):
        # output -> probabilities
        return None

    def to_classes(self, ypred, method=None):
        # output -> classes.
        # note: only in rare cases this method is overloaded (e.g., for methods
        # that do not produce probabilities or if the method has a special way
        # of computing classes).
        # 'method' parameter can be:
        # None   = allow the ordinal method to choose the best (or use argmax)
        # "mode" = class with highest probability (argmax)
        # "mean" = expectation average of the probabilities distribution
        # "median" = median weighted by the probabilities distribution
        assert method in (None, 'mode', 'mean', 'median')
        probs = self.to_probabilities(ypred)
        if method == 'mean':  # also called expectation trick by Beckham (2016)
            K = probs.shape[1]
            kk = torch.arange(K, device=probs.device, dtype=torch.float32)[None]
            return torch.round(torch.sum(kk * probs, 1)).long()
        elif method == 'median':
            # the weighted median is the value whose cumulative probability is 0.5
            Pc = torch.cumsum(probs, 1)
            return torch.sum(Pc < 0.5, 1)
        else:  # default=mode
            return probs.argmax(1)

    def to_scores(self, ypred):
        # output -> scalar rank score. by default, the output (if single output)
        # is the expected value from the probabilities (float).
        if self.how_many_outputs() == 1:
            return ypred[:, 0]
        device = ypred.device
        probs = self.to_probabilities(ypred)
        kk = torch.arange(self.K, device=ypred.device, dtype=torch.float32)[None]
        return torch.sum(kk * probs, 1)

################################################################################
# Classical losses.                                                            #
################################################################################

class CE(OrdinalMethod):
    def compute_loss(self, ypred, ytrue):
        return F.cross_entropy(ypred, ytrue, reduction='none')

    def to_probabilities(self, ypred):
        return F.softmax(ypred, dim=1)

CrossEntropy = CE  # TEMP

class MAE(OrdinalMethod):
    def how_many_outputs(self):
        return 1

    def compute_loss(self, ypred, ytrue):
        ypred = torch.clamp(ypred, 0, self.K-1)[:, 0]
        return torch.abs(ypred-ytrue)

    def to_classes(self, ypred, method=None):
        ypred = torch.round(ypred).long()[:, 0]
        ypred = torch.clamp(ypred, 0, self.K-1)
        return ypred

class MSE(MAE):
    def compute_loss(self, ypred, ytrue):
        ypred = torch.clamp(ypred, 0, self.K-1)[:, 0]
        return (ypred-ytrue)**2

class DummyMedian(OrdinalMethod):
    # This is just a useful hack to easily get a baseline benchmark. Always
    # returns the average class.
    def compute_loss(self, ypred, ytrue):
        return 0*ypred

    def to_classes(self, ypred, method=None):
        return (self.K//2) * torch.ones(len(ypred), dtype=torch.long, device=ypred.device)

################################################################################
# Cheng, Jianlin, Zheng Wang, and Gianluca Pollastri. "A neural network        #
# approach to ordinal regression." 2008 IEEE international joint conference on #
# neural networks (IEEE world congress on computational intelligence). IEEE,   #
# 2008. https://arxiv.org/pdf/0704.1028.pdf                              #
################################################################################
# Notice that some authors cite later papers like OR-CNN (Zhenxing Niu et al,  #
# 2016) but we believe this was the first for neural networks and is based on  #
# the Frank & Hall (2001) ordinal ensemble.                                    #
################################################################################

class OrdinalEncoding(OrdinalMethod):
    def how_many_outputs(self):
        return self.K-1

    def compute_loss(self, ypred, ytrue):
        # if K=4, then
        #                k = 0  1  2
        #     Y=0 => P(Y>k)=[0, 0, 0]
        #     Y=1 => P(Y>k)=[1, 0, 0]
        #     Y=2 => P(Y>k)=[1, 1, 0]
        #     Y=3 => P(Y>k)=[1, 1, 1]
        KK = torch.arange(self.K-1, device=ytrue.device).expand(len(ytrue), -1)
        yytrue = (ytrue[:, None] > KK).float()
        return torch.sum(F.binary_cross_entropy_with_logits(ypred, yytrue, reduction='none'), 1)

    def to_probabilities(self, ypred):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y>k) into P(Y=k)
        # P(Y=0) = 1-P(Y>0)
        # P(Y=1) = P(Y>0)-P(Y>1)
        # ...
        # P(Y=K-1) = P(Y>K-2)
        probs = torch.sigmoid(ypred)
        prob_0 = 1-probs[:, [0]]
        prob_k = probs[:, [-1]]
        probs = torch.cat((prob_0, probs[:, :-1]-probs[:, 1:], prob_k), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return probs

    def to_classes(self, ypred, method=None):
        # for OrdinalEncoding, if method=None (default) use the cumulative
        # distribution directly to get the classes, as suggested in the paper.
        if method is None:
            # notice we are working on the logit space, therefore yp>0 is the
            # same as sigmoid(yp)>0.5
            return torch.sum(ypred >= 0, 1)
        return super().to_classes(ypred, method)

################################################################################
# Barbero-Gómez, Javier, Pedro Antonio Gutiérrez, and César Hervás-Martínez.   #
# "Error-correcting output codes in the framework of deep ordinal              #
# classification." Neural Processing Letters (2022): 1-32.                     #
# https://link.springer.com/article/10.1007/s11063-022-10824-7                 #
################################################################################

class ECOC(OrdinalEncoding):
    def compute_loss(self, ypred, ytrue):
        # if K=4, then
        #                k = 0  1  2
        #     Y=0 => P(Y>k)=[0, 0, 0]
        #     Y=1 => P(Y>k)=[1, 0, 0]
        #     Y=2 => P(Y>k)=[1, 1, 0]
        #     Y=3 => P(Y>k)=[1, 1, 1]
        ypred = torch.sigmoid(ypred)
        KK = torch.arange(self.K-1, device=ytrue.device).expand(len(ytrue), -1)
        yytrue = (ytrue[:, None] > KK).float()
        return torch.sum((ypred - yytrue)**2, 1)

################################################################################
# McCullagh, Peter. "Regression models for ordinal data." Journal of the Royal #
# Statistical Society: Series B (Methodological) 42.2 (1980): 109-127.         #
# https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1980.tb01109.x
################################################################################
# This work precedes OrdinalEncoding, but it is similar, except that weights   #
# are shared; only biases are different. Furthermore, it learns P(Y≤j).        #
# P(Y≤j|x) = sigmoid(θⱼ - βx)                                                  #
################################################################################

class POM(OrdinalMethod):
    def __init__(self, K):
        super().__init__(K)
        biases = torch.zeros(1, self.K-1)
        self.biases = torch.nn.parameter.Parameter(biases)

    def how_many_outputs(self):
        return 1

    def compute_loss(self, ypred, ytrue):
        ypred = self.biases - ypred
        KK = torch.arange(self.K-1, device=ytrue.device).expand(len(ytrue), -1)
        yytrue = (ytrue[:, None] <= KK).float()
        return torch.sum(F.binary_cross_entropy_with_logits(ypred, yytrue, reduction='none'), 1)

    def to_proba(self, ypred):
        # P(Y=k) = P(Y≤k)-P(Y≤k-1)
        ypred = self.biases - ypred
        probs = torch.sigmoid(ypred)
        last_probs = 1-probs[:, [-1]]
        probs[:, 1:] = probs[:, 1:]-probs[:, 0:-1]
        probs = torch.cat((probs, last_probs), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return probs

    def to_classes(self, ypred, method=None):
        if method is None:
            # if none, use the biases as thresholds in the logit space
            return torch.bucketize(ypred[:, 0], self.biases[0])
        probs = self.to_proba(ypred)
        return super().to_classes(probs, method)

################################################################################
# Vargas, Victor Manuel, Pedro Antonio Gutiérrez, and César Hervás-Martínez.   #
# "Cumulative link models for deep ordinal classification." Neurocomputing     #
# 401 (2020): 48-58.                                                           #
# https://www.sciencedirect.com/science/article/pii/S0925231220303805          #
################################################################################

class CumulativeLinkLoss(POM):
    def __init__(self, K, link_function=torch.sigmoid, init_cutpoints='ordered'):
        super().__init__(K)
        assert init_cutpoints in ('ordered', 'random')
        self.link_function = link_function
        ncutpoints = self.K-1
        if init_cutpoints == 'ordered':
            cutpoints = torch.arange(ncutpoints, dtype=torch.float32) \
                - ncutpoints/2
        else:
            cutpoints = torch.rand(ncutpoints).sort()[0]
        self.cutpoints = torch.nn.parameter.Parameter(cutpoints)

    def compute_loss(self, ypred, ytrue):
        probs = self.to_probabilities(ypred)
        each = torch.arange(len(ytrue), device=ytrue.device)
        return -torch.log(probs[each, ytrue]+1e-7)  # cross-entropy

    def to_probabilities(self, ypred):
        ypred = self.link_function(self.cutpoints - ypred)
        link_mat = ypred[:, 1:] - ypred[:, :-1]
        return torch.cat((ypred[:, [0]], link_mat, 1-ypred[:, [-1]]), 1)

################################################################################
# Cao, Wenzhi, Vahid Mirjalili, and Sebastian Raschka. "Rank consistent        #
# ordinal regression for neural networks with application to age               #
# estimation." Pattern Recognition Letters 140 (2020): 325-331.                #
# https://www.sciencedirect.com/science/article/pii/S016786552030413X          #
################################################################################
# Very similar to POM (McCullagh, 1980), but extends OrdinalEncoding.          #
################################################################################

class CORAL(OrdinalEncoding):
    # CORAL works by having the same weights but different biases - to make it
    # simple to users, we will ask for 1 output and then insert the K-1 biases
    # ourselves.
    def __init__(self, K, preinit_bias=True):
        super().__init__(K)
        # preinit_bias is based on the author's github
        if preinit_bias:
            biases = torch.arange(self.K-1, 0, -1, dtype=torch.float32) / (self.K-1)
        else:
            biases = torch.zeros(self.K-1)
        self.biases = torch.nn.parameter.Parameter(biases[None])

    def how_many_outputs(self):
        return 1

    def compute_loss(self, ypred, ytrue):
        ypred = ypred + self.biases
        return super().compute_loss(ypred, ytrue)

    def to_probabilities(self, ypred):
        ypred = ypred + self.biases
        return super().to_probabilities(ypred)

    def to_classes(self, ypred, method=None):
        ypred = ypred + self.biases
        return super().to_classes(ypred, method)

################################################################################
# Shi, Xintong, Wenzhi Cao, and Sebastian Raschka. "Deep neural networks for   #
# rank-consistent ordinal regression based on conditional probabilities."      #
# Pattern Analysis and Applications 26.3 (2023): 941-955.                      #
# https://link.springer.com/article/10.1007/s10044-023-01181-9                 #
################################################################################
# This work extends OrdinalEncoding but it ensures that the output is          #
# consistent since it uses conditionality, i.e., it multiplies P(Y>k+1) by     #
# P(Y>k), therefore P(Y>k+1) is necessarily smaller than P(Y>k).               #
################################################################################

class CORN(OrdinalMethod):
    def how_many_outputs(self):
        return self.K-1

    def compute_loss(self, ypred, ytrue):
        # implementation-wise, the only difference is the subset variable. this
        # is equivalent to what's done in the paper: use subset outputs ≥ K
        KK = torch.arange(self.K-1, device=ytrue.device).expand(len(ytrue), -1)
        yytrue = (ytrue[:, None] > KK).float()
        subset = (ytrue[:, None] >= KK).float()
        return torch.sum(subset * F.binary_cross_entropy_with_logits(ypred, yytrue, reduction='none'), 1)

    def to_probabilities(self, ypred):
        # probabilities are conditional
        sigmoids = torch.sigmoid(ypred)
        prob0 = 1-sigmoids[:, [0]]
        return torch.cat((prob0, torch.cumprod(sigmoids, 1)), 1)

    def to_classes(self, ypred, method=None):
        if method is None:
            probs = torch.cumprod(torch.sigmoid(ypred), 1)
            return torch.sum(probs >= 0.5, 1)
        return super().to_classes(ypred, method)

################################################################################
# Fernandes, Kelwin, and Jaime S. Cardoso. "Ordinal image segmentation using   #
# deep neural networks." 2018 International Joint Conference on Neural         #
# Networks (IJCNN). IEEE, 2018.                                                #
# https://ieeexplore.ieee.org/abstract/document/8489527                        #
################################################################################

class HardOrdinal(OrdinalMethod):
    # the version from Kelwin paper is slightly different than CORN: there is no
    # subset, and the probabilities are < (inferior), not > (superior).
    def how_many_outputs(self):
        return self.K-1

    def compute_loss(self, ypred, ytrue):
        # model returns conditional "corrected" probabilities
        # convert the conditional "corrected" probabilities to "corrected" probabilities
        sigmoids = torch.sigmoid(ypred)
        probs_plus = torch.cumprod(sigmoids, 1)
        # the loss now operates on these "corrected" probabilities
        KK = torch.arange(self.K-1, device=ytrue.device).expand(len(ytrue), -1)
        yytrue = (ytrue[:, None] > KK).float()
        return torch.sum(F.binary_cross_entropy(probs_plus, yytrue, reduction='none'), 1)

    def to_probabilities(self, ypred):
        # model returns conditional "corrected" probabilities
        # convert the conditional "corrected" probabilities to "corrected" probabilities
        sigmoids = torch.sigmoid(ypred)
        probs_plus = torch.cumprod(sigmoids, 1)
        # convert the "corrected" probabilities to probabilities
        prob_0 = 1-probs_plus[:, [0]]
        prob_k = probs_plus[:, [-1]]
        probs = torch.cat((prob_0, probs_plus[:, :-1]-probs_plus[:, 1:], prob_k), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return probs

    def to_classes(self, ypred, method=None):
        if method is None:
            # model returns conditional "corrected" probabilities
            # convert the conditional "corrected" probabilities to "corrected" probabilities
            sigmoids = torch.sigmoid(ypred)
            probs_plus = torch.cumprod(sigmoids, 1)
            return torch.sum(probs_plus >= 0.5, 1)
        return super().to_classes(ypred, method)

class HardOrdinalParallel(HardOrdinal):
    # This version is similar to CORAL.
    def __init__(self, K):
        super().__init__(K)
        biases = torch.zeros(1, self.K-1)
        self.biases = torch.nn.parameter.Parameter(biases)

    def how_many_outputs(self):
        return 1

    def compute_loss(self, ypred, ytrue):
        return super().compute_loss(ypred+self.biases, ytrue)

    def to_probabilities(self, ypred):
        return super().to_probabilities(ypred+self.biases)

################################################################################
# da Costa, Joaquim F. Pinto, Hugo Alonso, and Jaime S. Cardoso. "The unimodal #
# model for the classification of ordinal data." Neural Networks 21.1 (2008):  #
# 78-91.                                                                       #
# https://www.sciencedirect.com/science/article/pii/S089360800700202X          #
################################################################################

class BinomialUnimodal_CE(OrdinalMethod):
    def how_many_outputs(self):
        return 1

    def compute_loss(self, ypred, ytrue):
        log_probs = self.to_log_probabilities(ypred)
        return F.nll_loss(log_probs, ytrue, reduction='none')

    def to_probabilities(self, ypred):
        # it is numerically better to operate in the log-space due to precision
        # overflows introduced by the factorial.
        return torch.exp(self.to_log_probabilities(ypred))

    def to_log_probabilities(self, ypred):
        # used internally by the loss in compute_loss()
        device = ypred.device
        log_probs = F.logsigmoid(ypred)
        log_inv_probs = F.logsigmoid(-ypred)
        N = ypred.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = log_fact(K-1) + kk*log_probs + (K-kk-1)*log_inv_probs
        den = log_fact(kk) + log_fact(K-kk-1)
        return num - den

class BinomialUnimodal_MSE(BinomialUnimodal_CE):
    def compute_loss(self, ypred, ytrue):
        device = ypred.device
        probs = self.to_probabilities(ypred)
        yonehot = torch.zeros(probs.shape[0], self.K, device=device)
        yonehot[range(probs.shape[0]), ytrue] = 1
        return torch.sum((probs - yonehot)**2, 1)

################################################################################
# Beckham, Christopher, and Christopher Pal. "A simple squared-error           #
# reformulation for ordinal classification." arXiv preprint arXiv:1612.00775   #
# (2016). https://arxiv.org/pdf/1612.00775.pdf                                 #
################################################################################

class HiddenSoftmax_fixA(OrdinalMethod):
    def compute_loss(self, ypred, ytrue):
        a = torch.arange(self.K, device=ypred.device, dtype=torch.float32)[None]
        ypred = torch.sum(a * F.softmax(ypred, 1), 1)
        return (ypred-ytrue)**2

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

    def to_classes(self, ypred, method=None):
        # authors recommend that 'mean' be used
        if method == None: method = 'mean'
        return super().to_classes(ypred, method)

class HiddenSoftmax_learnA(OrdinalMethod):
    # unbounded version: predicted class = sum(a*softmax(output))
    def __init__(self, K):
        super().__init__(K)
        a = 0.01*torch.randn(1, self.K, dtype=torch.float32)
        self.a = torch.nn.parameter.Parameter(a)

    def compute_loss(self, ypred, ytrue):
        ypred = torch.sum(self.a * F.softmax(ypred, 1), 1)
        return (ypred-ytrue)**2

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

    def to_classes(self, ypred, method=None):
        if method == None:
            ypred = torch.round(torch.sum(self.a * F.softmax(ypred, 1), 1))
            ypred = torch.clamp(ypred, 0, self.K-1)
            return ypred.long()
        return super().to_classes(ypred, method)

class HiddenSoftmax_learnAsigm(HiddenSoftmax_learnA):
    # bounded version: predicted class = (K-1)*sigmoid(sum(a*softmax(output)))
    def compute_loss(self, ypred, ytrue):
        ypred = torch.sum(self.a * F.softmax(ypred, 1), 1)
        ypred = (self.K-1)*torch.sigmoid(ypred)  # ensure [0,K-1]
        return (ypred-ytrue)**2

    def to_classes(self, ypred, method=None):
        if method == None:
            ypred = torch.sum(self.a * F.softmax(ypred, 1), 1)
            ypred = torch.round((self.K-1)*torch.sigmoid(ypred))
            return ypred.long()
        return super().to_classes(ypred, method)

################################################################################
# Beckham, Christopher, and Christopher Pal. "Unimodal probability             #
# distributions for deep ordinal classification." International Conference on  #
# Machine Learning. PMLR, 2017.                                                #
# http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf                   #
################################################################################

class PoissonUnimodal(OrdinalMethod):
    def __init__(self, K, tau=1, learn_tau=False):
        # tau controls the variance of the distribution
        # the given tau is ignored if learn_tau=True
        super().__init__(K)
        self.learn_tau = learn_tau
        if learn_tau:
            self.tau = torch.tensor(0, dtype=torch.float32)
            self.tau = torch.nn.parameter.Parameter(self.tau)
        else:
            self.tau = torch.tensor(tau, dtype=torch.float32)

    def how_many_outputs(self):
        return 1

    def compute_loss(self, ypred, ytrue):
        return F.cross_entropy(self.activation(ypred), ytrue, reduction='none')

    def to_probabilities(self, ypred):
        return F.softmax(self.activation(ypred), 1)

    def to_classes(self, ypred, method=None):
        # authors recommend that 'mean' be used
        if method == None: method = 'mean'
        return super().to_classes(ypred, method)

    def activation(self, ypred):
        # internal function used by compute_loss() and to_probabilities()
        # they apply softplus (relu) to avoid log(negative)
        ypred = F.softplus(ypred)
        KK = torch.arange(1., self.K+1, device=ypred.device)[None]
        h = KK*torch.log(ypred+1e-6) - ypred - log_fact(KK)
        # apply sigmoid if tau is learnable
        tau = torch.sigmoid(self.tau) if self.learn_tau else self.tau
        return h / tau

class PoissonUnimodalLearnTau(PoissonUnimodal):
    def __init__(self, K):
        super().__init__(K, learn_tau=True)

################################################################################
# Yamasaki, Ryoya. "Unimodal Likelihood Models for Ordinal Data." Transactions #
# on Machine Learning Research, 2022                                           #
# https://openreview.net/forum?id=1l0sClLiPc                                   #
################################################################################

# Basically:
# ORD-ACL: probabilities=ACL(ORD(X @ β))
# VS-SL: probabilities=SOFTMAX(-VS(ORD(X @ β)))
# where ORD transforms logits so its ascending like a stair
# (logits[0] + cumsum(logits[1:]**2)), ACL normalizes probabilities with adjacents
# (P(Y)/(P(Y)+P(Y+1)), VS forces logits to look like an inverted V shape (-abs(logits))

class ORD_ACL(OrdinalMethod):
    def how_many_outputs(self):
        return self.K-1

    def to_probabilities(self, logits, logprobs=False):
        zeros = torch.zeros(len(logits), 1, device=logits.device)
        # ORD
        ǵ = logits[:, [0]] + torch.cat((zeros, torch.cumsum(torch.exp(logits[:, 1:]), 1)), 1)
        # ACL
        u = torch.cat((zeros, torch.cumsum(ǵ, 1)), 1)
        return F.log_softmax(-u, 1) if logprobs else torch.softmax(-u, 1)

    def compute_loss(self, ypred, ytrue):
        ypred = self.to_proba(ypred, True)
        return F.nll_loss(ypred, ytrue)

class VS_SL(OrdinalMethod):
    def to_probabilities(self, logits, logprobs=False):
        zeros = torch.zeros(len(logits), 1, device=logits.device)
        # ORD
        ǵ = logits[:, [0]] + torch.cat((zeros, torch.cumsum(torch.exp(logits[:, 1:]), 1)), 1)
        # VS
        ǧ = ǵ**2  # or torch.abs(ǵ)
        return F.log_softmax(-ǧ, 1) if logprobs else torch.softmax(-ǧ, 1)

    def compute_loss(self, ypred, ytrue):
        ypred = self.to_proba(ypred, True)
        return F.nll_loss(ypred, ytrue)

################################################################################
# Araújo, Teresa, et al. "DR| GRADUATE: Uncertainty-aware deep learning-based  #
# diabetic retinopathy grading in eye fundus images." Medical Image Analysis   #
# 63 (2020): 101715.                                                           #
# https://www.sciencedirect.com/science/article/pii/S1361841520300797          #
################################################################################

class GaussianUncertainty(OrdinalMethod):
    def __init__(self, K, alpha=0.7):
        # the default alpha comes from the paper
        super().__init__(K)
        assert 0 <= alpha <= 1
        self.alpha = alpha

    def how_many_outputs(self):
        return 2

    def compute_loss(self, ypred, ytrue):
        logprobs = torch.log(self.to_probabilities(ypred)+1e-6)
        sigma2 = ypred[:, 1]**2
        return self.alpha*F.nll_loss(logprobs, ytrue, reduction='none') + \
            (1-self.alpha)*sigma2

    def to_probabilities(self, ypred):
        # not sure how the authors ensure that average is in [0,K)
        # we have used a sigmoid
        avg = torch.sigmoid(ypred[:, [0]])*(self.K-1)
        # we need a relative large epsilon for sigma2 to avoid infinity
        sigma2 = ypred[:, [1]]**2 + 0.01
        i = torch.arange(self.K, device=ypred.device)[None]
        sqrt_2pi = 2.5066282746310002
        probs = (1/(sqrt_2pi*sigma2)) * torch.exp(-0.5*((i-avg)**2)/sigma2)
        probs = probs / probs.sum(1, True)
        return probs

################################################################################
# de La Torre, Jordi, Domenec Puig, and Aida Valls. "Weighted kappa loss       #
# function for multi-class classification of ordinal data in deep learning."   #
# Pattern Recognition Letters 105 (2018): 144-154.                             #
# https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666      #
################################################################################
# Use n=2 (default) for Quadratic Weighted Kappa.                              #
# Notice that the other losses are reduction='none'. But this loss, by its     #
# very nature, always returns a scalar.                                        #
################################################################################

class WeightedKappa(OrdinalMethod):
    def __init__(self, K, n=2):
        super().__init__(K)
        self.n = 2

    def compute_loss(self, ypred, ytrue):
        probs = F.softmax(ypred, 1)
        kk = torch.arange(self.K, device=ytrue.device)
        i, j = torch.meshgrid(kk, kk, indexing='xy')
        w = torch.abs(i-j)**self.n
        N = torch.sum(w[ytrue] * probs)
        probs_sum = torch.sum(probs, 0)
        D = sum((torch.sum(ytrue == i)/len(ytrue)) * torch.sum(w[i] * probs_sum) for i in range(self.K))
        kappa = 1 - N/D
        return torch.log(1-kappa+1e-7)

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

QuadraticWeightedKappa = WeightedKappa

################################################################################
# Cruz, Ricardo, et al. "Ordinal class imbalance with ranking." Iberian        #
# conference on pattern recognition and image analysis. Springer, Cham, 2017.  #
# https://link.springer.com/chapter/10.1007/978-3-319-58838-4_1                #
################################################################################
# Fernandes, Kelwin, Jaime S. Cardoso, and Birgitte Schmidt Astrup. "A deep    #
# learning approach for the forensic evaluation of sexual assault." Pattern    #
# Analysis and Applications 21.3 (2018): 629-640.                              #
# https://link.springer.com/article/10.1007/s10044-018-0694-3                  #
################################################################################
# scoring pairwise ranknet produces scores. the authors then convert to classes
# by choosing thresholds. another possibility the authors could have considered
# (which would have been nicer to implement) would be to compute a rolling
# average score per class during training, and then predicting each class based
# on minimum distance to the average scores.

class OrdinalRankNet(OrdinalMethod):
    def __init__(self, K, cost_strategy='absolute'):
        super().__init__(K)
        assert cost_strategy in ('homogeneous', 'absolute', 'inverse')
        self.cost_strategy = cost_strategy
        self.register_buffer('thresholds', torch.zeros(K-1))
        self.scores = []
        self.labels = []

    def how_many_outputs(self):
        return 1

    def compute_loss(self, ypred, ytrue):
        ypred = ypred[:, 0]
        self.scores.append(torch.sigmoid(ypred))
        self.labels.append(ytrue)
        ytrue = (torch.sign(ytrue[None] - ytrue[:, None]) + 1) / 2
        ypred = ypred[None] - ypred[:, None]
        return F.binary_cross_entropy_with_logits(ypred, ytrue)

    def post_epoch(self):
        if len(self.scores) == 0: return
        scores = torch.cat(self.scores)
        labels = torch.cat(self.labels)
        self.thresholds = self.decide_thresholds(scores, labels, self.K, self.cost_strategy)
        self.scores = []
        self.labels = []

    def to_classes(self, ypred, method=None):
        assert method is None
        ypred = torch.sigmoid(ypred[:, 0])
        return torch.sum(ypred[:, None] >= self.thresholds[None], 1)

    @staticmethod
    def decide_thresholds(scores, labels, nclasses, cost_strategy):
        import sys
        sys.setrecursionlimit(max(1000, len(scores)*2))
        memo = {}
        if cost_strategy == 'uniform':
            cost = 1-torch.eye(nclasses)
        elif cost_strategy == 'absolute':
            ii = torch.arange(nclasses)
            xx, yy = torch.meshgrid(ii, ii, indexing='xy')
            cost = torch.abs(xx-yy)
        elif cost_strategy == 'inverse':
            count = torch.bincount(labels, minlength=nclasses)
            cost = len(labels) / (nclasses*count+1)

        def f(k, i):
            if i == len(scores):
                return 0, []
            if (k, i) in memo:
                return memo[(k, i)]
            # returns tuple (cost, threshold list)
            err = cost[k, labels[i]].item()
            if k+1 >= nclasses:
                c, t = f(k, i+1)
                c += err
            else:
                c1, t1 = f(k, i+1)
                c1 += err
                c2, t2 = f(k+1, i)
                c = min(c1, c2)
                midscore = scores[0] if i == 0 else (scores[i-1]+scores[i])/2
                t = t1 if c1 < c2 else [midscore] + t2
            memo[(k, i)] = (c, t)
            return c, t

        # we need to sort by scores before calling "f"
        ix = torch.argsort(scores)
        scores = scores[ix]
        labels = labels[ix]
        ths = f(0, 0)[1]
        # if there are missing classes, then we need to fill up thresholds
        if len(ths) < nclasses-1:
            inf = torch.tensor(float('inf'), device=scores.device)
            ths += [inf]*((nclasses-1)-len(ths))
        return torch.stack(ths)

################################################################################
# Liu, Yanzhu, Adams Wai Kin Kong, and Chi Keong Goh. "A constrained deep      #
# neural network for ordinal regression." Proceedings of the IEEE conference   #
# on computer vision and pattern recognition. 2018.                            #
# https://ieeexplore.ieee.org/document/8578191                                 #
################################################################################
# this loss avoids having to convert scores as the previous loss by doing
# multi-task: it predicts both scores and classification. then ignores scores.

class CNNPOR(OrdinalMethod):
    def __init__(self, K, C=1):
        super().__init__(K)
        self.C = C

    def how_many_outputs(self):
        # this model requires an extra output (ignored on testing) for the
        # pairwise hinge loss
        return self.K + 1

    def compute_loss(self, ypred, ytrue):
        # in order to make it compatible and fair with the other methods, we
        # use the randomly batches directly. one disadvantage is there is no
        # garantee how many "d" adjacent ranks exist in each batch.
        Gc = ypred[:, :-1]
        Gr = ypred[:, -1]
        # l1 = softmax logistic loss
        l1 = F.cross_entropy(ypred, ytrue, reduction='none')
        # l2 = pairwise hinge loss
        zero = torch.zeros((), device=ypred.device)
        l2 = 0
        for k in range(self.K-1):
            kk = ytrue == k
            kk1 = ytrue == k+1
            if kk.sum() > 0 and kk1.sum() > 0:
                Ok = Gr[kk]
                Ok1 = Gr[kk1]
                l2 += torch.sum(torch.maximum(zero, 1 + Ok[None] - Ok1[:, None]))
        return l1 + self.C*l2

    def to_probabilities(self, ypred):
        # ignore the last output which is only used for training
        return F.softmax(ypred[:, :-1], 1)

################################################################################
# Chu, Wei, and S. Sathiya Keerthi. "New approaches to support vector ordinal  #
# regression." Proceedings of the 22nd international conference on Machine     #
# learning. 2005. https://dl.acm.org/doi/abs/10.1145/1102351.1102370           #
################################################################################
# Adapted for logistic loss here:                                              #
# Rennie, Jason DM, and Nathan Srebro. "Loss functions for preference levels:  #
# Regression with discrete ordered labels." Proceedings of the IJCAI           #
# multidisciplinary workshop on advances in preference handling. Vol. 1. AAAI  #
# Press, Menlo Park, CA, 2005.                                                 #
# https://people.csail.mit.edu/jrennie/papers/ijcai05-preference.pdf           #
################################################################################

# using logistic as the loss
# SVORIM = all thresholds
class SVORIM(OrdinalMethod):
    def __init__(self, K):
        super().__init__(K)
        self.thetas = torch.nn.parameter.Parameter(torch.zeros(K-1))

    def how_many_outputs(self):
        return 1

    def compute_loss(self, ypred, ytrue):
        kk = torch.arange(self.K-1, device=ypred.device)[None]
        s = 2*(kk >= ytrue[:, None])-1
        loss = -torch.log(torch.sigmoid(s*(self.thetas[None] - ypred)))
        return torch.sum(loss, 1)

    def to_classes(self, ypred, method=None):
        return torch.sum(ypred >= self.thetas[None], 1)

# SVOREX = immediate (adjacent) thresholds
class SVOREX(SVORIM):
    def compute_loss(self, ypred, ytrue):
        kk = torch.arange(self.K-1, device=ypred.device)[None]
        s = -1*(kk == ytrue[:, None]-1) + 1*(kk == ytrue[:, None])
        loss = -torch.log(torch.sigmoid(s*(self.thetas[None] - ypred)))
        return torch.sum(loss, 1)

################################################################################
# Fathony, Rizal, Mohammad Ali Bashiri, and Brian Ziebart. "Adversarial        #
# surrogate losses for ordinal regression." Advances in Neural Information     #
# Processing Systems 30 (2017).                                                #
# https://proceedings.neurips.cc/paper/2017/hash/c86a7ee3d8ef0b551ed58e354a836f2b-Abstract.html #
################################################################################

class AdversarialOrdTh(SVORIM):
    def compute_loss(self, ypred, ytrue):
        kk = torch.arange(1, self.K+1, dtype=torch.float32, device=ypred.device)[None]
        zero = torch.zeros([1], device=ypred.device)
        # Σ_{k>=j} \theta_k  =>  [θ₁+θ₂+θ₃; θ₂+θ₃; θ₃; 0]  => [flip(cumsum(flip(θ))); 0]
        sums = torch.cat((torch.flip(torch.cumsum(torch.flip(self.thetas, [0]), 0), [0]), zero))
        term1 = torch.amax((kk*(ypred + 1) + sums[None])/2, 1)
        term2 = torch.amax((kk*(ypred - 1) + sums[None])/2, 1)
        term3 = (ytrue+1)*ypred[:, 0]
        term4 = sums[ytrue]
        return term1 + term2 - term3 - term4

class AdversarialOrdMc(OrdinalMethod):
    def compute_loss(self, ypred, ytrue):
        kk = torch.arange(1, self.K+1, dtype=torch.float32, device=ypred.device)[None]
        term1 = torch.amax((ypred + kk) / 2, 1)
        term2 = torch.amax((ypred - kk) / 2, 1)
        term3 = torch.gather(ypred, 1, ytrue[:, None])[:, 0]
        return term1 + term2 - term3

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

################################################################################
# Liu, Xiaofeng, et al. "Unimodal regularized neuron stick-breaking for        #
# ordinal classification." Neurocomputing 388 (2020): 34-44.                   #
# https://www.sciencedirect.com/science/article/pii/S0925231220300618          #
################################################################################

class NeuronStickBreaking(OrdinalMethod):
    def how_many_outputs(self):
        return self.K-1

    def activation(self, ypred):
        P = torch.sigmoid(ypred)
        ones = torch.ones(len(P), 1, device=P.device)
        invcum_P = torch.cumprod(1-P, 1)
        return torch.cat((P, ones), 1) * torch.cat((ones, invcum_P), 1)

    def compute_loss(self, ypred, ytrue):
        ypred = self.activation(ypred)
        if len(ytrue.shape) == 1:
            ytrue = F.one_hot(ytrue, self.K).float()
        return torch.sum(F.binary_cross_entropy(ypred, ytrue, reduction='none'), 1)

    def to_probabilities(self, ypred):
        return self.activation(ypred)

class UnimodalRegularization(OrdinalMethod):
    def uniform(self, i, y): return 1/K
    def poisson(self, i, y): return (((y+1)**i)*torch.exp(-(y+1))/fact(i)) / torch.sum(((y+1)**i)*torch.exp(-(y+1))/fact(i), 1)
    # for binomial, not sure what they have done, but I am taking
    # p=(y+1)/(K+1) so that the mean is y
    def binomial(self, i, y): return (fact(K)/(fact(i)*fact(K-i))) * ((y+1)/(K+1)) * ((1-(y+1)/(K+1))**(K-i))
    def exp(self, i, y): return torch.softmax(-torch.abs(i-y)/self.tau, 1)

    # unimodal label smoothing technique, proposed by NeuronStickBreaking, but
    # can be used by other ordinal methods, such as CrossEntropy
    # q(i, l) = (1-eta)*onehot(i, l) + eta*f(i, l)
    # where i=class index, l=ground-truth, f=smoothness function
    def __init__(self, K, ordinal_method=NeuronStickBreaking, f='exp', eta=0.15, tau=1):
        super().__init__(K)
        assert 0 <= eta <= 1
        self.ordinal_method = ordinal_method(K)
        self.f = getattr(UnimodalRegularization, f)
        self.eta = eta
        self.tau = tau

    def how_many_outputs(self):
        return self.ordinal_method.how_many_outputs()

    def compute_loss(self, ypred, ytrue):
        ii = torch.arange(self.K, device=ypred.device)[None]
        yy = ytrue[:, None]
        delta = (ii == yy).float()
        ytrue = self.eta*delta + (1-self.eta)*self.f(self, ii, yy)
        return self.ordinal_method.compute_loss(ypred, ytrue)

    def to_probabilities(self, ypred):
        return self.ordinal_method.to_probabilities(ypred)

class CrossEntropy_UR(UnimodalRegularization):
    # convenience class to test CE w/ UR
    def __init__(self, K, f='exp', eta=0.15, tau=1):
        super().__init__(K, CrossEntropy, f, eta, tau)

################################################################################
# Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Ordinal losses for   #
# classification of cervical cancer risk." PeerJ Computer Science 7 (2021):    #
# e457. https://peerj.com/articles/cs-457/                                     #
################################################################################
# These losses require two parameters: omega and lambda.                       #
# The default omega value comes from the paper.                                #
# The default lambda values comes from our experiments.                        #
################################################################################

def entropy_term(ypred):
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    P = F.softmax(ypred, 1)
    logP = F.log_softmax(ypred, 1)
    return -torch.sum(P * logP, 1)

def neighbor_term(ypred, ytrue, margin):
    margin = torch.tensor(margin, device=ytrue.device)
    P = F.softmax(ypred, 1)
    K = P.shape[1]
    dP = torch.diff(P, dim=1)
    sign = (torch.arange(K-1, device=ytrue.device)[None] >= ytrue[:, None])*2-1
    return torch.sum(F.relu(margin + sign*dP), 1)

class CO2(OrdinalMethod):
    def __init__(self, K, lamda=0.01, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def compute_loss(self, ypred, ytrue):
        term = neighbor_term(ypred, ytrue, self.omega)
        return F.cross_entropy(ypred, ytrue, reduction='none') + self.lamda*term

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

class CO(CO2):
    # CO is the same as CO2 with omega=0
    def __init__(self, K, lamda=0.01):
        super().__init__(K, lamda, 0)

class HO2(OrdinalMethod):
    def __init__(self, K, lamda=1.0, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def compute_loss(self, ypred, ytrue):
        term = neighbor_term(ypred, ytrue, self.omega)
        return entropy_term(ypred) + self.lamda*term

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

################################################################################
# Albuquerque, Tomé, Ricardo Cruz, and Jaime S. Cardoso. "Quasi-Unimodal       #
# Distributions for Ordinal Classification." Mathematics 10.6 (2022): 980.     #
# https://www.mdpi.com/2227-7390/10/6/980                                      #
################################################################################
# These losses require two parameters: omega and lambda.                       #
# The default omega value comes from the paper.                                #
# The default lambda values comes from our experiments.                        #
################################################################################

def quasi_neighbor_term(ypred, ytrue, margin):
    margin = torch.tensor(margin, device=ytrue.device)
    P = F.softmax(ypred, 1)
    K = P.shape[1]
    ix = torch.arange(len(P))

    # force close neighborhoods to be inferior to True class prob
    has_left = ytrue > 0
    close_left = has_left * F.relu(margin+P[ix, ytrue-1]-P[ix, ytrue])
    has_right = ytrue < K-1
    close_right = has_right * F.relu(margin+P[ix, (ytrue+1)%K]-P[ix, ytrue])

    # force distant probabilities to be inferior than close neighborhoods of true class
    left = torch.arange(K, device=ytrue.device)[None] < ytrue[:, None]-1
    distant_left = torch.sum(left * F.relu(margin+P-P[ix, ytrue-1][:, None]), 1)
    right = torch.arange(K, device=ytrue.device)[None] > ytrue[:, None]+1
    distant_right = torch.sum(right * F.relu(margin+P-P[ix, (ytrue+1)%K][:, None]), 1)

    return close_left + close_right + distant_left + distant_right

class QUL_CE(OrdinalMethod):
    def __init__(self, K, lamda=0.1, omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def compute_loss(self, ypred, ytrue):
        term = quasi_neighbor_term(ypred, ytrue, self.omega)
        return F.cross_entropy(ypred, ytrue, reduction='none') + self.lamda*term

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

class QUL_HO(OrdinalMethod):
    def __init__(self, K, lamda=10., omega=0.05):
        super().__init__(K)
        self.lamda = lamda
        self.omega = omega

    def compute_loss(self, ypred, ytrue):
        term = quasi_neighbor_term(ypred, ytrue, self.omega)
        return entropy_term(ypred) + self.lamda*term

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

################################################################################
# Castagnos, François, Martin Mihelich, and Charles Dognin. "A Simple Log-     #
# -based Loss Function for Ordinal Text Classification." Proceedings of the    #
# 29th International Conference on Computational Linguistics. 2022.            #
# https://aclanthology.org/2022.coling-1.407.pdf                               #
################################################################################
# Polat, Gorkem, et al. "Class Distance Weighted Cross-Entropy Loss for        #
# Ulcerative Colitis Severity Estimation." arXiv preprint arXiv:2202.05167     #
# (2022). https://arxiv.org/pdf/2202.05167.pdf                                 #
################################################################################
# These two papers propose something identical. Not sure which paper came      #
# first. Interestingly, CDW_CE recommends alpha=5, while OrdinalLogLoss        #
# recommends alpha=1.5 (which for us also works better).                       #
################################################################################

class OrdinalLogLoss(OrdinalMethod):
    def __init__(self, K, alpha=1.5):
        super().__init__(K)
        self.alpha = alpha

    def d(self, y):
        # internal function for the distance penalization. you may overload
        # this function if you want to use another.
        i = torch.arange(self.K, device=y.device)
        return torch.abs(i[None] - y[:, None])**self.alpha

    def compute_loss(self, ypred, ytrue):
        ypred = F.softmax(ypred, 1)
        return -torch.sum(torch.log(torch.clip(1-ypred, 1e-6)) * self.d(ytrue), 1)

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

class CDW_CE(OrdinalLogLoss):
    def __init__(self, K, alpha=5):
        super().__init__(K, alpha)

################################################################################
# Som, Anirudh, et al. "A machine learning approach to assess student group    #
# collaboration using individual level behavioral cues." Computer Vision–ECCV  #
# 2020 Workshops: Glasgow, UK, August 23–28, 2020, Proceedings, Part VI 16.    #
# Springer International Publishing, 2020.                                     #
# https://link.springer.com/chapter/10.1007/978-3-030-65414-6_8                #
################################################################################

class OrdinalCrossEntropy(OrdinalMethod):
    def compute_loss(self, ypred, ytrue):
        w = torch.abs(ypred.argmax(1) - ytrue)
        return (1+w) * F.cross_entropy(ypred, ytrue, reduction='none')

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

################################################################################
# Cardoso, Jaime S., Ricardo Cruz, and Tomé Albuquerque. "Unimodal             #
# Distributions for Ordinal Regression." arXiv preprint arXiv:2303.04547       #
# (2023). https://arxiv.org/abs/2303.04547                                     #
################################################################################

class UnimodalNet(OrdinalMethod):
    def compute_loss(self, ypred, ytrue):
        return F.cross_entropy(self.activation(ypred), ytrue, reduction='none')

    def to_probabilities(self, ypred):
        return F.softmax(self.activation(ypred), 1)

    def activation(self, ypred):
        # force outputs to be positive
        # for differentiable reasons, we use softplus instead of relu
        relu = F.softplus
        ypred = relu(ypred)
        # if output=[X,Y,Z] => lr_slope=[X,X+Y,X+Y+Z]
        # if output=[X,Y,Z] => rl_slope=[Z,Z+Y,Z+Y+X]
        lr_slope = torch.cumsum(ypred, 1)
        rl_slope = torch.flip(torch.cumsum(torch.flip(ypred, [1]), 1), [1])
        ypred = torch.minimum(lr_slope, rl_slope)
        return ypred

def unimodal_wasserstein(p, mode):
    # Returns the closest unimodal distribution to p with the given mode.
    # Return tuple:
    # 0: total transport cost
    # 1: closest unimodal distribution
    import numpy as np
    from scipy.spatial.distance import squareform, pdist
    from scipy.optimize import linprog
    assert abs(p.sum()-1) < 1e-6, 'Expected normalized probability mass.'
    assert np.any(p >= 0), 'Expected nonnegative probabilities.'
    assert len(p.shape) == 1, 'Probabilities p must be a vector.'
    assert 0 <= mode < p.size, 'Invalid mode value.'
    K = p.size
    C = squareform(pdist(np.arange(K)[:, None]))  # cost matrix
    Ap = [([0]*i + [1] + [0]*(K-i-1))*K for i in range(K)]
    Ai = [[0]*i*K + [1]*K + [-1]*K + [0]*(K-i-2)*K if i < mode else
          [0]*i*K + [-1]*K + [1]*K + [0]*(K-i-2)*K for i in range(K-1)]
    result = linprog(C.ravel(), A_ub=Ai, b_ub=np.zeros(K-1), A_eq=Ap, b_eq=p)
    T = result.x.reshape(K, K)
    return (T*C).sum(), T.sum(1)

def emd(p, q):
    # https://en.wikipedia.org/wiki/Earth_mover%27s_distance
    pp = p.cumsum(1)
    qq = q.cumsum(1)
    return torch.sum(torch.abs(pp-qq), 1)

def is_unimodal(p):
    # checks (true/false) whether the given probability vector is unimodal. this
    # function is not used by the following classes, but it is used in the paper
    # to compute the "% times unimodal" metric
    zero = torch.zeros(1, device=p.device)
    p = torch.sign(torch.round(torch.diff(p, prepend=zero, append=zero), decimals=2))
    p = torch.diff(p[p != 0])
    p = p[p != 0]
    return len(p) <= 1

class WassersteinUnimodal_KLDIV(OrdinalMethod):
    def __init__(self, K, lamda=100.):
        super().__init__(K)
        self.lamda = lamda

    def compute_loss(self, ypred, ytrue):
        probs = F.softmax(ypred, 1)
        probs_log = F.log_softmax(ypred, 1)
        closest_unimodal = torch.stack([
            torch.tensor(unimodal_wasserstein(phat, y)[1], dtype=torch.float32, device=ytrue.device)
            for phat, y in zip(probs.cpu().detach().numpy(), ytrue.cpu().numpy())])
        term = self.distance_loss(probs, probs_log, closest_unimodal)
        return F.cross_entropy(ypred, ytrue, reduction='none') + self.lamda*term

    def distance_loss(self, phat, phat_log, target):
        return torch.sum(F.kl_div(phat_log, target, reduction='none'), 1)

    def to_probabilities(self, ypred):
        return F.softmax(ypred, 1)

class WassersteinUnimodal_Wass(WassersteinUnimodal_KLDIV):
    def distance_loss(self, phat, phat_log, target):
        return emd(phat, target)
