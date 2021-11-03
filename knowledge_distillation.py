import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse
from contextualized_topic_models.models.ctm import CTM
import pickle, os
from tqdm import tqdm
from utils import load_model

def get_posteriors(teacher_dataset, teacher_model, batch_size=25, num_workers=10):
    tp = pickle.load(open(os.path.join(teacher_model, "tp.pkl"), "rb"))
    ctm = load_model(teacher_model, len(tp.vocab))
    
    ctm.model.zero_grad()

    posterior_variances = []
    posterior_means = []
    posterior_log_variances = []

    loader = DataLoader(teacher_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch_samples in tqdm(loader):
        X_bow = batch_samples['X_bow']
        X_bow = X_bow.reshape(X_bow.shape[0], -1)
        X_contextual = batch_samples['X_contextual']
        prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance, word_dists, \
            estimated_labels =\
                ctm.model(X_bow.cuda(), X_contextual.cuda())
        posterior_variances.append(posterior_variance)
        posterior_means.append(posterior_mean)
        posterior_log_variances.append(posterior_log_variance)

        posterior_variances = torch.cat(posterior_variances)
        posterior_means = torch.cat(posterior_means)
        posterior_log_variances = torch.cat(posterior_log_variances)

        return posterior_variances, posterior_means, posterior_log_variances



class CTMDatasetPosteriors(Dataset):

    """Class to load BoW and the contextualized embeddings."""

    def __init__(self, X_contextual, X_bow, idx2token, posterior_variance, posterior_mean, posterior_log_variance):

        if X_bow.shape[0] != len(X_contextual):
            raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                            "You might want to check if the BoW preparation method has removed some documents. ")

        self.X_bow = X_bow
        self.X_contextual = X_contextual
        self.idx2token = idx2token        

        self.posterior_variance = posterior_variance
        self.posterior_mean = posterior_mean
        self.posterior_log_variance = posterior_log_variance
        

    def __len__(self):
        """Return length of dataset."""
        return self.X_bow.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        if type(self.X_bow[i]) == scipy.sparse.csr.csr_matrix:
            X_bow = torch.FloatTensor(self.X_bow[i].todense())
            X_contextual = torch.FloatTensor(self.X_contextual[i])
        else:
            X_bow = torch.FloatTensor(self.X_bow[i])
            X_contextual = torch.FloatTensor(self.X_contextual[i])

        posterior_variance = torch.FloatTensor(self.posterior_variance[i])
        posterior_mean = torch.FloatTensor(self.posterior_mean[i])
        posterior_log_mean = torch.FloatTensor(self.posterior_log_mean[i])
        

        return_dict = {'X_bow': X_bow, 'X_contextual': X_contextual,
                       'posterior_variance' : posterior_variance,
                       'posterior_mean': posterior_mean,
                       'posterior_log_mean': posterior_log_mean}


        return return_dict


        

class StudentZeroShotTM(CTM):
    def __init__(self, **kwargs):
        inference_type = "zeroshot"
        super().__init__(**kwargs, inference_type=inference_type)
        


    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        print("Using teacher posterior")

        for batch_samples in loader:
            # batch_size x vocab_size
            X_bow = batch_samples['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples['X_contextual']

            teacher_posterior_variance = batch_samples['posterior_variance']
            teacher_posterior_mean = batch_samples['posterior_mean']
            teacher_posterior_log_mean = batch_samples['posterior_log_mean']

            
            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()
                posterior_variance = posterior_variance.cuda()
                posterior_mean = posterior_mean.cuda()
                posterior_log_mean = posterior_log_mean.cuda()


            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists, estimated_labels = self.model(X_bow, X_contextual, labels)

            # backward pass
            kl_loss, rl_loss = self._loss(
                X_bow, word_dists, teacher_posterior_mean, teacher_posterior_variance,
                posterior_mean, posterior_variance, posterior_log_variance)

            loss = self.weights["beta"]*kl_loss + rl_loss
            loss = loss.sum()

            if labels is not None:
                target_labels = torch.argmax(labels, 1)

                label_loss = torch.nn.CrossEntropyLoss()(estimated_labels, target_labels)
                loss += label_loss

            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X_bow.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss
