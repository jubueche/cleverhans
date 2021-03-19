from absl import app, flags

import datasets
import itertools
import time
import jax.numpy as np
import numpy.random as npr
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import logsoftmax

from cleverhans.jax.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.jax.attacks.projected_gradient_descent import projected_gradient_descent

import ujson as json
from copy import deepcopy
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS


def main(_):
    rng = random.PRNGKey(0)

    # Load MNIST dataset
    train_images, train_labels, test_images, test_labels = datasets.mnist()

    batch_size = 128
    batch_shape = (-1, 28, 28, 1)
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    train_images = np.reshape(train_images, batch_shape)
    test_images = np.reshape(test_images, batch_shape)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    def save(fn,opt_state):
        params = deepcopy(get_params(opt_state))
        save_dict = {}
        for idx,p in enumerate(params):
            if(p != ()):
                pp = (p[0].tolist(),p[1].tolist())
                params[idx] = pp
        save_dict["params"] = params
        with open(fn, "w") as f:
            json.dump(save_dict, f)

    def load(fn):
        with open(fn, "r") as f:
            params = json.load(f)
        params = params["params"]
        for idx,p in enumerate(params):
            if(p != []):
                pp = (np.array(p[0]),np.array(p[1]))
                params[idx] = pp
            else:
                params[idx] = ()
        return opt_init(params)

    batches = data_stream()

    # Model, loss, and accuracy functions
    init_random_params, predict = stax.serial(
        stax.Conv(32, (8, 8), strides=(2, 2), padding="SAME"),
        stax.Relu,
        stax.Conv(128, (6, 6), strides=(2, 2), padding="VALID"),
        stax.Relu,
        stax.Conv(128, (5, 5), strides=(1, 1), padding="VALID"),
        stax.Flatten,
        stax.Dense(128),
        stax.Relu,
        stax.Dense(10),
    )

    def loss(params, batch):
        inputs, targets = batch
        preds = predict(params, inputs)
        return -np.mean(logsoftmax(preds) * targets)

    def accuracy(params, batch):
        inputs, targets = batch
        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(predict(params, inputs), axis=1)
        return np.mean(predicted_class == target_class)

    def gen_ellipsoid(X, zeta_rel, zeta_const, alpha, N_steps):
        zeta = (np.abs(X).T * zeta_rel).T + zeta_const
        if(alpha is None):
            alpha = 1/N_steps * zeta
        else:
            assert isinstance(alpha,float), "Alpha must be float"
            alpha = alpha*np.ones_like(X)
        return zeta,alpha

    def gen_ellipsoid_match_volume(X, zeta_const, eps, alpha, N_steps):
        x_norms = np.linalg.norm(np.reshape(X,(X.shape[0],-1)) , ord=1, axis=1)
        N = np.prod(X.shape[1:])
        zeta_rel = N*(eps-zeta_const) / x_norms
        assert (zeta_rel <= 1.0).all(), "Zeta rel cannot be larger than 1. Please increase zeta const or reduce eps"
        zeta_rel = np.clip(0.0, zeta_rel, 1.0)
        return gen_ellipsoid(X, zeta_rel, zeta_const, alpha, N_steps)

    # Instantiate an optimizer
    opt_init, opt_update, get_params = optimizers.adam(0.001)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    # Initialize model
    _, init_params = init_random_params(rng, batch_shape)
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    try:
        opt_state = load("tutorials/jax/test_model.json")
    except:
        # Training loop
        print("\nStarting training...")
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time
        save("tutorials/jax/test_model.json", opt_state)

    # Evaluate model on clean data
    params = get_params(opt_state)

    # Evaluate model on adversarial data
    model_fn = lambda images: predict(params, images)
    # Generate single attacking test image
    idx = 0
    plt.figure(figsize=(15,6),constrained_layout=True)
    
    # zeta, alpha = gen_ellipsoid(X=test_images[idx].reshape((1,28,28,1)), zeta_rel=FLAGS.zeta_rel, zeta_const=FLAGS.zeta_const, alpha=None, N_steps=40)
    zeta, alpha = gen_ellipsoid_match_volume(X=test_images[idx].reshape((1,28,28,1)), zeta_const=FLAGS.zeta_const, eps=FLAGS.eps, alpha=None, N_steps=40)
    test_images_pgd_ellipsoid = projected_gradient_descent(model_fn, test_images[idx].reshape((1,28,28,1)), zeta, alpha, 40, np.inf)

    test_images_fgm = fast_gradient_method(model_fn, test_images[idx].reshape((1,28,28,1)), FLAGS.eps, np.inf)
    
    test_images_pgd = projected_gradient_descent(model_fn, test_images[idx].reshape((1,28,28,1)), FLAGS.eps, 0.01, 40, np.inf)

    plt.subplot(141)
    plt.imshow(np.squeeze(test_images[idx]),cmap='gray')
    plt.title("Original")
    plt.subplot(142)
    plt.imshow(np.squeeze(test_images_fgm),cmap='gray')
    plt.title("FGM")
    plt.subplot(143)
    plt.imshow(np.squeeze(test_images_pgd),cmap='gray')
    plt.title("PGD")
    plt.subplot(144)
    plt.imshow(np.squeeze(test_images_pgd_ellipsoid),cmap='gray')
    plt.title("PGD Ellipsoid")
    plt.show()

    # Generate whole attacking test images
    # zeta, alpha = gen_ellipsoid(X=test_images, zeta_rel=FLAGS.zeta_rel, zeta_const=FLAGS.zeta_const, alpha=None, N_steps=40)
    zeta, alpha = gen_ellipsoid_match_volume(X=test_images, zeta_const=FLAGS.zeta_const, eps=FLAGS.eps, alpha=None, N_steps=40)
    test_images_pgd_ellipsoid = projected_gradient_descent(model_fn, test_images, zeta, alpha, 40, np.inf)
    test_acc_pgd_ellipsoid = accuracy(params, (test_images_pgd_ellipsoid, test_labels))

    test_images_fgm = fast_gradient_method(model_fn, test_images, FLAGS.eps, np.inf)
    test_images_pgd = projected_gradient_descent(model_fn, test_images, FLAGS.eps, 0.01, 40, np.inf)
     
    test_acc_fgm = accuracy(params, (test_images_fgm, test_labels))
    test_acc_pgd = accuracy(params, (test_images_pgd, test_labels))

    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))

    print("Training set accuracy: {}".format(train_acc))
    print("Test set accuracy on clean examples: {}".format(test_acc))
    print("Test set accuracy on FGM adversarial examples: {}".format(test_acc_fgm))
    print("Test set accuracy on PGD adversarial examples: {}".format(test_acc_pgd))
    print("Test set accuracy on PGD Ellipsoid adversarial examples: {}".format(test_acc_pgd_ellipsoid))


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_float("eps", 0.035, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_float("zeta_const", 0.01, "Constant offset of ellipsoid.")
    flags.DEFINE_float("zeta_rel", 0.25, "Relative offset of ellipsoid.")

    app.run(main)
