import tensorflow as tf
import fire
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.training.input import batch

import model, sample, encoder
import model
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        print("hparams :",hparams,'\n',
              "length :", length,'\n',
              "context :", context,'\n',
              "batch_size :",batch_size,'\n',
              "temperature :", temperature,'\n',
              "top_k :",top_k,'\n',
              "top_p :",top_p,'\n')
        '''
        hparams : [('n_ctx', 1024), ('n_embd', 768), ('n_head', 12), ('n_layer', 12), ('n_vocab', 50257)] 
        length : 512 
        context : Tensor("Placeholder:0", shape=(1, ?), dtype=int32)
        batch_size : 1
        temperature : 1
        top_k : 0
        top_p : 1
        '''

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)
if __name__ == '__main__':
    fire.Fire(interact_model)
################################################################################

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        print(logits)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
###########################################################################

def top_k_logits(logits, k):
    if k == 0:
        # no 절단
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        print(min_values)
        # tf.newaxis : size(차원) 변경

        return tf.where(
            # tf.where(bool type 텐서, true일 때 출력값, false일 때 출력값)
            # x, y가 없으면 참 요소의 좌표(2D 텐서)를 반환한다. 
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10, # -100억
            # tf.ones_like : 모든 요소가 1로 설정된 tensor와 동일한 유형 및 모양의 tensor를 리턴한다.
            logits, 
        )

    # tf.cond :  tf.equal이면 logits 동작이 실행되고, _top_k()는 실행되지 않는다.
    return tf.cond( 
       tf.equal(k, 0), # k == 0 
       lambda: logits,
       lambda: _top_k(),
    )



def top_p_logits(logits, p):
    """핵심 sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1) 
    # 내림차순으로 logits을 정렬, sort() = sort(axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    # tf.cumsum : 누적 합계를 수행 (ex. ([a, b, c])   # [a, a + b, a + b + c] )
    indices = tf.stack([ 
        # indices = index의 복수
        # tf.stack : (a,b,c)shape에서 N텐서의 길이가 주어졌을 때, axis=0이면 (n,a,b,c) / axis = 1 이면 (a,n,b,c)가 된다
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
        # cast :  텐서를 새로운 형태로 캐스팅하는데 사용한다.
        #         cumulative_probs가 p보다 작거나 같도록 하여 boolean형태로 나타낸다
        # reduce_sum : 텐서의 차원들을 탐색하며 개체들의 총합을 계산한다. 
        #              cast에서 나온 값에서 -1을 해주고 열 단위로 더해준다.
        # tf.maximum : 최댓값 반환
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        # start_token이 none인 경우
        # start token 이나 context 중 하나를 정확히 지정해야 한다. 
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
        # [batch size, 1] shape에서 start_token으로 다 채워준다.
        print("냐냐 :", context)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
        # reuse=tf.AUTO_REUSE : 변수가 없는 경우 변수를 생성하고 그렇지 않은 경우 반환한다.
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'): # 이름 범위
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            # tf.multinomial : 다항분포로부터 샘플을 뽑아준다.
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            # *args : 여러개의 인자를 함수에 전달할 때 쓰인다.
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens

sample_sequence(
hparams= [('n_ctx', 1024), ('n_embd', 768), ('n_head', 12), ('n_layer', 12), ('n_vocab', 50257)], 
length= 512,
context= tf.Tensor("Placeholder:0", shape=(1, )), 
batch_size = 1,
temperature = 1,
top_k = 0,
top_p = 1)