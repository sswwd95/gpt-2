import tensorflow as tf

import model

def top_k_logits(logits, k):
    if k == 0:
        # no 절단
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
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
