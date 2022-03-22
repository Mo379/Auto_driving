import jax 
import jax.numpy as jnp
import numpy as np

def init_model(model_shape, parent_key, scale = 0.01):
    """ Initialise the MLP model
    This function initialises mlp model parameters (weights and biases)

    Args:
        model_shape: input list
        parent_key: input PRNGKey
        scale: input float
    """
    params = []
    keys = jax.random.split(parent_key, num=len(model_shape)-1)
    for in_dim, out_dim, key in zip(model_shape[:-1],model_shape[1:],keys):
        w_key, b_key = jax.random.split(key)
        params.append([
                scale*jax.random.normal(w_key, shape=(out_dim,in_dim)),
                scale*jax.random.normal(b_key, shape=(out_dim,))
            ])
    return params

def forward(params, x):
    """Predict method
    This function predicts output value of the network for an input

    Args: 
        params: input pytree (trained model)
        x: input array
    """
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(jnp.dot(layer[0],x) + layer[1])
    return jnp.dot(last[0], x) + last[1]
batch_forward = jax.vmap(forward, in_axes = (None,0))

def loss_fn(params, x, y):
    predictions = batch_forward(params,x)
    return (1/len(x))*jnp.sum((predictions-y)**2)

@jax.jit
def RMSprop_update(params, x, y,lr, r_t):
    """ RMSprob optimised full pass
    A forward and backwards pass over the network parameters using RMS prob as the optimiser

    Args:
        params: input pytree (untrained model)
        x: input array (training examples)
        lr: input float (learning rate)
        r_t: input pytree (optimiser_state)
    """
    loss,grads = jax.value_and_grad(loss_fn)(params,x,y)
    params,r_t = rms_prop(params,grads,r_t)
    return r_t,loss,params


def rms_prop(params,grad,r_,alpha=0.0001,rho=0.9,epsilon=10**(-7)):
    """RMSprop, stateful optimiser
    This function computes the ouput of the RMSprop algorithm

    Args:
        params: input pytree
        grad: input pytree
        r_: input pytree
    """
    r_t = jax.tree_multimap(lambda g,r: rho*r +(1-rho)*g**(2) ,grad,r_)
    w_t = jax.tree_multimap(lambda g,r,p: p - alpha*(1/(jnp.sqrt(r)+epsilon))*g,grad,r_t,params)
    return w_t,r_t












