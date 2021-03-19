import jax.numpy as np

from cleverhans.jax.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.jax.utils import clip_eta, one_hot

import numpy as onp
from scipy.optimize import minimize

def projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=None,
    rand_minmax=0.3,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to 0. or the
    Madry et al. (2017) method when rand_minmax is larger than 0.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :return: a tensor for the adversarial example
    """
    BALL = isinstance(eps,float) 
    assert (BALL or eps.shape == x.shape), "Eps must define an epsilon ball or an ellipsoid"
    assert (not BALL) or np.array(eps_iter <= eps).all(), (eps_iter, eps)
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")

    # Initialize loop variables
    if rand_init:
        rand_minmax = eps
        eta = np.random.uniform(x.shape, -rand_minmax, rand_minmax)
        eta = clip_eta(eta, norm, eps)
    else:
        eta = np.zeros_like(x)

    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        x_labels = np.argmax(model_fn(x), 1)
        y = one_hot(x_labels, 10)

    for _ in range(nb_iter):
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        if not isinstance(eps, float) and (norm == 2):
            raise NotImplementedError
            # Projection onto the ellipsoid in l2
            """
            perturbation = Proj(x + grads) - x
            Optimization problem: x_{proj}* = arg min_{x_proj} .5 * ||x_{proj}-y||_2^2 s.t. (x_{proj}-c)' W (x_{proj}-c) <= 1 
            """
            adv_x *= eps_iter
            x_ = x.ravel()
            y_ = adv_x.ravel() # We want to project y back on the ellipsoid defined by eps
            w_ = 1 / (eps.ravel() ** 2 + 1e-12) # Squared inverse of the diagonal matrix W that transforms the ball into an axis-aligned ellipsoid
            def f_and_g(x_p):
                g_ = x_p - y_
                f_ = .5 * np.linalg.norm(g_, ord=2) ** 2
                return f_, g_
            def functionValIneq(x_p):
                t0 = x_p - x_
                return np.dot(t0, w_ * t0)
            def gradientIneq(x_p):
                t0 = x_p - x_
                return 2*(w_*t0)

            x0 = onp.random.randn(x_.shape[0])
            bnds = [(-np.inf,np.inf)] * x_.shape[0]
            constraints = ({'type' : 'ineq',
                        'fun' : lambda x: -functionValIneq(x),
                        'jac' : lambda x: -gradientIneq(x)})
            result = minimize(f_and_g, x0, jac=True, method='SLSQP',
                            bounds=bnds,
                            constraints=constraints)
            x_p = result.x
            print("\nFunction value ineq.:",functionValIneq(x_p))
            adv_x = np.reshape(x_p, x.shape)

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)

    return adv_x
