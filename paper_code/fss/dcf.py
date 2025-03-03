# This file implements FSS functions described in fig1 of https://eprint.iacr.org/2020/1392.pdf#page=11.15
# https://www.youtube.com/watch?v=Zm-MUVve2_w
from typing import List

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# we choose Group G = Z_N, where N = 2 ^ n. We will fix n = 64.

GROUP_BIT_NUM = 64


def keys_to_new_randoms(six_key: jnp.array, dtype: any) -> List:
    """Convert six keys to  new randoms"""
    return [
        jax.random.bits(six_key[0], dtype=dtype),
        int(jax.random.bernoulli(six_key[1])),
        int(jax.random.bernoulli(six_key[2])),
        jax.random.bits(six_key[3], dtype=dtype),
        int(jax.random.bernoulli(six_key[4])),
        int(jax.random.bernoulli(six_key[5])),
    ]


# we use jax uint32 array to represent lambda-length bit string array
def G(random_bits: jnp.array, unpack=True) -> jnp.array:
    """Let G : {0, 1}^λ → {0, 1}^2(λ+2) be a pseudorandom generator.
    We take lambda = 32 and use uint32 to represent the bit string array.

    input shape (1,) uintlambda jnp array
    output [uintlambda jnp array, bool, bool, uintlambda jnp array, bool, bool]
    """

    # convert byte array into keys
    key = jax.random.key(random_bits)

    # split keys into 6
    six_keys = jax.random.split(key, 6)

    # generate 6 new random bit arrays of shape lambda | lambda | 1 | lambda | lambda | 1
    return keys_to_new_randoms(six_keys, random_bits.dtype)


def get_nth_bit(num, n):
    """Gets the nth bit of an unsigned integer.

    Args:
      num: The unsigned integer.
      n: The bit position (0-indexed) (Little Endian).

    Returns:
      The value of the nth bit (0 or 1).
    """

    return (num >> n) & 1


def bit_decompose(x: jnp.uint64) -> List[int]:
    """convert an integer to a list of bits"""
    return [get_nth_bit(x, i) for i in range(GROUP_BIT_NUM - 1, -1, -1)]


def print_byte_in_binary(byte_value):
    """for debug, print byte in binary"""
    binary_representation = format(byte_value, '08b')
    print(binary_representation)


def flip(zero_or_one: int):
    """flip a bit"""
    return 1 - zero_or_one


def DCF_gen(alpha: jnp.uint64, random_key=jax.random.key(1212)):
    """Distributed Comparison Function generation
    outputs 1 if x < α and 0 otherwise.
    Assuming two party xor share everything
    """
    lamb_type = jnp.uint64
    alpha_bits = bit_decompose(alpha)
    random_key, new_random_key = jax.random.split(random_key, 2)
    s0_0 = jax.random.bits(random_key, dtype=lamb_type)
    s1_0 = jax.random.bits(new_random_key, dtype=lamb_type)

    # paper is indexed from 1 to 64, but we start from 0
    s0_last = s0_0
    s1_last = s1_0

    b0_last = 0
    b1_last = 1

    CWs = []
    for i in range(GROUP_BIT_NUM):
        s0_L, b0_L, c0_L, s0_R, b0_R, c0_R = G(s0_last)
        s1_L, b1_L, c1_L, s1_R, b1_R, c1_R = G(s1_last)

        if alpha_bits[i] == 0:
            s0_Keep = s0_L
            s1_Keep = s1_L
            b0_Keep = b0_L
            b1_Keep = b1_L
            s0_Lose = s0_R
            s1_Lose = s1_R
        else:
            s0_Keep = s0_R
            s1_Keep = s1_R
            b0_Keep = b0_R
            b1_Keep = b1_R
            s0_Lose = s0_L
            s1_Lose = s1_L

        sCW = jnp.bitwise_xor(s0_Lose, s1_Lose)
        bCW_L = b0_L ^ b1_L ^ alpha_bits[i] ^ 1
        bCW_R = b0_R ^ b1_R ^ alpha_bits[i]
        cCW_L = c0_L ^ c1_L ^ alpha_bits[i]
        cCW_R = c0_R ^ c1_R
        CW = [sCW, bCW_L, bCW_R, cCW_L, cCW_R]

        CWs.extend(CW)

        if alpha_bits[i] == 0:
            bCW_Keep = bCW_L
        else:
            bCW_Keep = bCW_R

        s0_last = jnp.bitwise_xor(s0_Keep, b0_last * sCW)
        b0_last = b0_Keep ^ (b0_last * bCW_Keep)
        s1_last = jnp.bitwise_xor(s1_Keep, b1_last * sCW)
        b1_last = b1_Keep ^ (b1_last * bCW_Keep)

    CWs.extend(CW)
    k0 = [s0_0] + CWs
    k1 = [s1_0] + CWs
    return (k0, k1)


def DCF_eval(b: int, kb: List, x: jnp.uint64):
    """eval DCF

    Args:
        b (int): must be 0 or 1
        kb (List): k0 or k1, from DCF_gen
        x (jnp.uint64): input to function f : x -> f(x)
    """
    s = kb[0]
    CWs = kb[1:]
    x_bits = bit_decompose(x)
    b_last = b
    c = 0

    for i in range(GROUP_BIT_NUM):
        start = i * 5
        end = start + 5
        (sCW, bCW_L, bCW_R, cCW_L, cCW_R) = CWs[start:end]
        s_hat_L, b_hat_L, c_hat_L, s_hat_R, b_hat_R, c_hat_R = G(s)
        s_L = jnp.bitwise_xor(s_hat_L, b_last * sCW)
        b_L = b_hat_L ^ (b_last * bCW_L)
        c_L = c_hat_L ^ (b_last * cCW_L)
        s_R = jnp.bitwise_xor(s_hat_R, b_last * sCW)
        b_R = b_hat_R ^ (b_last * bCW_R)
        c_R = c_hat_R ^ (b_last * cCW_R)

        if x_bits[i] == 0:
            s = s_L
            b_last = b_L
            c ^= c_L
        else:
            s = s_R
            b_last = b_R
            c ^= c_R

    return (b_last + c) % 2


if __name__ == '__main__':
    # generate key for function <= 10 return 1, else 0
    k0, k1 = DCF_gen(10)

    eval0 = DCF_eval(0, k0, 1000)
    eval1 = DCF_eval(1, k1, 1000)

    result = eval0 ^ eval1
    assert result == 0, f"eval0 xor eval1 != 0,{result}"

    eval0 = DCF_eval(0, k0, 10)
    eval1 = DCF_eval(1, k1, 10)

    result = eval0 ^ eval1
    assert result == 1, f"eval0 xor eval1 != 1,{result}"

    eval0 = DCF_eval(0, k0, 11)
    eval1 = DCF_eval(1, k1, 11)

    result = eval0 ^ eval1
    assert result == 0, f"eval0 xor eval1 != 0,{result}"

    # performance_test
    import time

    start_time = time.time()
    eval0 = DCF_eval(0, k0, 11)
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.5f} seconds")
