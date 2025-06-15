from math import log
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI
from numpyro.infer import Trace_ELBO
import optax
from jax import random
import jax

from numpyro.infer.autoguide import AutoMultivariateNormal

from torch.utils.tensorboard import SummaryWriter

from numpy import save

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

from collections import defaultdict

import copy

from prettytable import PrettyTable


from tqdm import tqdm

class MNGMMClassifier():

    def __init__(self, num_dim, num_classes, with_early_stop):
        self.num_dim = num_dim
        self.num_classes = num_classes

        self.pca = None

        self.global_params = None

        self.label_offset = 0

        self.with_early_stop = with_early_stop


    def update_dir_infos(self, log_dir = "logs/", save_dir = "saved_models/"):
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir

    def init_parameters(self, n_epochs, lr, log_dir, save_dir, batch_size, increment=10, base=50, scaling_factor=1.2, use_correct_scaling_factor=True, early_stop_ratio=0):

        self.num_steps = n_epochs

        self.init_lr = lr

        self.batch_size = batch_size
        self.save_dir = save_dir

        self.writer = SummaryWriter(log_dir)

        self.increment = increment

        self.num_base = base

        self.scaling_factor = scaling_factor

        self.use_correct_scaling_factor = use_correct_scaling_factor

        self.early_stop_ratio = early_stop_ratio


    def model(self, X, y=None, num_classes=2, global_params=None, **kwargs):
        num_features = X.shape[1]

        if global_params is None:
            class_means = numpyro.param("class_means", jnp.zeros((num_classes, num_features)))
            class_covs = numpyro.param("class_covs", jnp.stack([jnp.eye(num_features)] * num_classes))
        else :
            class_means = numpyro.param("class_means", global_params["class_means"])
            class_covs = numpyro.param("class_covs", global_params["class_covs"])
        
        with numpyro.plate("batch", X.shape[0], subsample_size=self.batch_size) as ind:
            X_batch = X[ind]
            y_batch = y[ind] if y is not None else None
            
            if y_batch is not None:
                base_dist = dist.MultivariateNormal(class_means[y_batch], class_covs[y_batch])
                numpyro.sample("obs", base_dist, obs=X_batch)


    def run_inference(self, X, y, test_X, test_y, log_prefix="", use_correct_scaling_factor=False):

        init_lr = self.init_lr

        scheduler = optax.join_schedules(
            schedules=[
                optax.linear_schedule(init_value=init_lr, end_value=init_lr*10, transition_steps=100),
                optax.exponential_decay(init_value=init_lr*10, transition_steps=500, decay_rate=0.85),
            ],
            boundaries=[100]
        )

        self.guide = lambda *args, **kwargs: None

        print("Initializing model")

        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(scheduler))

        self.svi = SVI(self.model, guide=self.guide, optim=optimizer, loss=Trace_ELBO())

        self.svi_state = self.svi.init(random.PRNGKey(0), X = X, y= y, num_classes=self.num_classes, global_params=self.global_params)

        last_state = None

        for step in tqdm(range(self.num_steps)):

            early_stop_flag, dets = self.calculate_metrics_on_covariances(self.svi.get_params(self.svi_state), increment=self.increment, use_correct_scaling_factor=use_correct_scaling_factor)

            if(self.with_early_stop & early_stop_flag & (last_state is not None) & (step > 1/3 * self.num_steps)):

                self.svi_state = last_state

                early_stop_flag, dets = self.calculate_metrics_on_covariances(self.svi.get_params(self.svi_state), increment=self.increment, use_correct_scaling_factor=use_correct_scaling_factor)
                correct, total, acc = self.calculate_acc(self.svi.get_params(self.svi_state), X, y)
                correct_test, total_test, acc_test = self.calculate_acc(self.svi.get_params(self.svi_state), test_X, test_y)

                self.writer.add_scalar(f"{log_prefix}/Accuracy/train", acc, step) 

                self.writer.add_scalar(f"{log_prefix}/Accuracy/test", acc_test, step)

                self.writer.add_scalar(f"{log_prefix}/LastCovariance/det_0", dets[0].item(), step)

                self.writer.add_scalar(f"{log_prefix}/Covariance/det_0", dets[1].item(), step)

                print(f"Step {step}: loss = {loss:.4f}, train_acc = {correct}/{total}, {acc:.2f}%,",
                      f" test_acc = {correct_test}/{total_test}, {acc_test:.2f}%, last_cov = {dets[0].item()}, cov = {dets[1].item()}, early_stop_flag = {early_stop_flag}")

                break

            last_state = self.svi_state

            self.svi_state, loss = self.svi.update(self.svi_state, X= X, y=y, num_classes=self.num_classes, covs_dets =dets)

            self.writer.add_scalar(f"{log_prefix}/Loss/train", loss.item(), step) 
            
            if step % 100 == 0:
                correct, total, acc = self.calculate_acc(self.svi.get_params(self.svi_state), X, y)
                correct_test, total_test, acc_test = self.calculate_acc(self.svi.get_params(self.svi_state), test_X, test_y)

                self.writer.add_scalar(f"{log_prefix}/Accuracy/train", acc, step) 

                self.writer.add_scalar(f"{log_prefix}/Accuracy/test", acc_test, step)

                self.writer.add_scalar(f"{log_prefix}/LastCovariance/det_0", dets[0].item(), step)

                self.writer.add_scalar(f"{log_prefix}/Covariance/det_1", dets[1].item(), step)

                print(f"Step {step}: loss = {loss:.4f}, train_acc = {correct}/{total}, {acc:.2f}%,",
                      f" test_acc = {correct_test}/{total_test}, {acc_test:.2f}%, last_cov = {dets[0].item()}, cov = {dets[1].item()}")

            if jnp.isnan(loss):
                print("Early stopping du to loss is NaN")
                self.svi_state = last_state
                break

            prev_loss = loss

        return self.svi.get_params(self.svi_state)


    def pre_processing(self, features, labels):
        if self.pca is None:
            self.pca = PCA(n_components=self.num_dim)
            features = self.pca.fit_transform(features)
        else:
            features = self.pca.transform(features)
        return features, labels


    def train(self, features, labels, test_features, test_labels, current_stage):
        features, labels = self.pre_processing(features, labels)
        test_features, test_labels = self.pre_processing(test_features, test_labels)

        labels = labels.astype(int)

        self.params = self.run_inference(jnp.array(features), jnp.array(labels), jnp.array(test_features), jnp.array(test_labels), log_prefix=f"stage_{current_stage}_Flearning", use_correct_scaling_factor=False)

        pred_labels, _ =self._predict(jnp.array(features), self.params)

        if self.global_params is not None:

            novel_idx = pred_labels >= self.label_offset

            print(f"Number of Novel Samples: {novel_idx.sum()} / {len(features)}")

            self.writer.add_scalar(f"Number/NovelSamples", novel_idx.sum().item(), current_stage)
            self.writer.add_scalar(f"Number/TotalSamples", len(features), current_stage)

            features = features[novel_idx]
            labels = labels[novel_idx]

            self.params = self.run_inference(jnp.array(features), jnp.array(labels), jnp.array(test_features), jnp.array(test_labels), log_prefix=f"stage_{current_stage}_Slearning", use_correct_scaling_factor=self.use_correct_scaling_factor)
        
        self.global_params = copy.deepcopy(self.params)


    def calculate_acc(self, params, test_features, test_labels):

        pred_test_labels, _ = self._predict(jnp.array(test_features), params)

        correct = jnp.sum(pred_test_labels == test_labels).tolist()
        
        return correct, len(test_features), 100. * (correct / float(len(test_features)))

    def test(self, test_features, test_labels):

        test_features, test_labels = self.pre_processing(test_features, test_labels)

        pred_test_labels, _ = self._predict(jnp.array(test_features), self.params)

        correct = jnp.sum(pred_test_labels == test_labels).tolist()

        return correct, len(test_features), 100. * correct / float(len(test_features))

    # output the acc of training data
    def _predict(self, X, params):
        class_means = params["class_means"]
        class_covs = params["class_covs"]
        log_probs = []

        for i in range(class_means.shape[0]):
            mvn = dist.MultivariateNormal(class_means[i], class_covs[i])
            log_probs.append(mvn.log_prob(X))
        
        log_probs = jnp.stack(log_probs, axis=-1)
        return jnp.argmax(log_probs, axis=-1), log_probs


    def _set_label_offset(self, label_offset):
        self.label_offset = label_offset

    def _correct_scaling_factors(self, n, total):
        return jnp.sqrt((n) / (total + n))

    def calculate_metrics_on_covariances(self, params, increment, use_correct_scaling_factor):
        class_covs = params["class_covs"]

        early_stop_flag = False

        if self.global_params is None:
            return early_stop_flag, [jnp.ones(1),
                                     jnp.ones(1)]

        global_class_covs = self.global_params["class_covs"]

        if not use_correct_scaling_factor:
            dets = [(jax.vmap(jnp.linalg.det)(global_class_covs[:self.num_base])).mean(), 
                (jax.vmap(jnp.linalg.det)(class_covs[self.label_offset: self.label_offset + increment])).mean()]
            scaling_factor = (self.label_offset + increment) / self.num_base

        else:
            dets = [(jax.vmap(jnp.linalg.det)(global_class_covs[:self.label_offset])).mean(), 
                (jax.vmap(jnp.linalg.det)(class_covs[self.label_offset: self.label_offset + increment])).mean()]
            scaling_factor = self._correct_scaling_factors(increment, self.label_offset)

        if(not jnp.isnan(dets[0])):
            if((dets[0] > 1) & (dets[1] > scaling_factor * dets[0])):
                early_stop_flag = True
            if((dets[0] < 1) & (dets[1] < dets[0] / scaling_factor)):
                early_stop_flag = True
        
        return early_stop_flag, dets

    def run(self, features, labels, test_features, test_labels, current_stage, testing_set):

        self.train(features, labels, test_features, test_labels, current_stage)

        t = PrettyTable(['TestSet','Correct', 'Smaples', 'Accuracy'])

        correct, total, acc = self.test(testing_set['test_all']._x, testing_set['test_all']._y)
        t.add_row(["All", correct, total, acc])
        self.writer.add_scalar(f"Test/Accuracy/All", acc, current_stage)

        correct, total, acc = self.test(testing_set['test_old']._x, testing_set['test_old']._y)
        t.add_row(["Old", correct, total, acc])
        self.writer.add_scalar(f"Test/Accuracy/Old", acc, current_stage)

        correct, total, acc = self.test(test_features, test_labels)
        t.add_row(["Novel", correct, total, acc])
        self.writer.add_scalar(f"Test/Accuracy/Novel", acc, current_stage)

        correct, total, acc = self.test(testing_set['known_test']._x, testing_set['known_test']._y)
        t.add_row(["S0",correct, total, acc])
        self.writer.add_scalar(f"Test/Accuracy/S0", acc, current_stage)

        print(t)

        # save the class means , covariances and supports to numpy files
        save(f"{self.save_dir}class_means.npy", np.array(self.params["class_means"]))
        save(f"{self.save_dir}class_covariances.npy", np.array(self.params["class_covs"]))

