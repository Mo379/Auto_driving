import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import operator as op
import functools
#
import numpy as np

#my libarary
def my_combinator(*layers):
  n_layers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fn(rng, input_shape):
    params = []
    for init_fun in init_funs:
      rng, layer_rng = jax.random.split(rng)
      input_shape, layer_params = init_fun(layer_rng, input_shape)#init funs return output-shape, params 
      params.append(layer_params)
    return input_shape, params
  def apply_fn(layers_params, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = jax.random.split(rng, n_layers) if rng is not None else (None,) * n_layers
    for apply_fun,params,rng in zip(apply_funs,layers_params,rngs):
      inputs = apply_fun(params,inputs, **kwargs)
    return inputs
  return init_fn, apply_fn



def element_wise(function, **fun_kwargs):
  init_fun = lambda rng,input_shape: (input_shape,())
  apply_fun = lambda params,inputs, **kwargs: function(inputs, **fun_kwargs)
  return init_fun, apply_fun
Relu_layer = element_wise(jax.nn.relu)
LeakyRelu_layer = element_wise(jax.nn.leaky_relu)
Softmax_layer = element_wise(jax.nn.softmax, axis=-1)
golrot = jax.nn.initializers.glorot_normal
normal = jax.nn.initializers.normal


def my_Dense(out_dim, w_init = golrot(), b_init = normal()):
  def init_fn(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k_w, k_b = jax.random.split(rng)
    w, b = w_init(k_w,(input_shape[-1], out_dim)),b_init(k_b,(out_dim,))
    params = (w,b)
    return output_shape, params


  def apply_fn(params, inputs, **kwargs):
    w,b = params
    return jnp.dot(inputs,w) + b
  return init_fn, apply_fn


def my_Flatten():
  def init_fn(rng, input_shape):
    output_shape = input_shape[0], functools.reduce(op.mul, input_shape[1:])
    return output_shape,  ()
  def apply_fn(params, inputs, **kwargs):
    return jnp.reshape(inputs, (inputs.shape[0], -1))
  return init_fn, apply_fn


