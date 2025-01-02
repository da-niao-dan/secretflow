# This file implements FSS functions described in https://eprint.iacr.org/2020/1392.pdf#page=11.15
from typing import List

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# we choose Group G = Z_N, where N = 2 ^ n. We will fix n = 64.

GROUP_BIT_NUM = 64


def six_key_to_array(six_key: jnp.array, dtype: any) -> jnp.array:
    """Convert six keys to six element array"""
    six_elements = jax.vmap(lambda x: jax.random.bits(x, dtype=dtype))(six_key)
    six_elements = six_elements.at[2].set(six_elements[2] % 2)
    six_elements = six_elements.at[5].set(six_elements[5] % 2)
    return six_elements


# we use jax uint32 array to represent lambda-length bit string array
def G(random_bits: jnp.array, unpack=True) -> jnp.array:
    """Let G : {0, 1}^λ → {0, 1}^2(2λ+1) be a pseudorandom generator.
    We take lambda = 32 and use uint32 to represent the bit string array.

    input shape (1,) uint32 jnp array
    output shape (1, 6) uint32 jnp array
    """

    # convert byte array into keys
    key = jax.random.key(random_bits)

    # split keys into 6
    six_keys = jax.random.split(key, 6)

    # generate 6 new random bit arrays of shape lambda | lambda | 1 | lambda | lambda | 1

    six_arrs = six_key_to_array(six_keys, random_bits.dtype)
    if unpack:
        return [six_arrs[i] for i in range(6)]
    else:
        return six_arrs


def convert_G(random_bits: jnp.array) -> jnp.uint64:
    """Convert bytes to jnp.uint64"""
    return random_bits.astype(jnp.uint64)


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


def DCF_gen(lamb: int, alpha: jnp.uint64, beta: jnp.uint64, key: int = 1212):
    """Distributed Comparison Function generation
    outputs β if x < α and 0 otherwise.

    """
    lamb_type = jnp.uint64 if lamb == 64 else jnp.uint32
    alpha_bits = bit_decompose(alpha)
    random_key = jax.random.key(key)
    random_key, new_random_key = jax.random.split(random_key, 2)
    s0_0 = jax.random.bits(random_key, dtype=lamb_type)
    s1_0 = jax.random.bits(new_random_key, dtype=lamb_type)
    Valpha = jnp.uint64(0)
    t0_0 = 0
    t1_0 = 1
    # paper is indexed from 1 to 64, but we start from 0
    s0_last = s0_0
    s1_last = s1_0
    t0_last = t0_0
    t1_last = t1_0

    CWs = []
    for i in range(GROUP_BIT_NUM):
        s0_L, v0_L, t0_L, s0_R, v0_R, t0_R = G(s0_last)
        s1_L, v1_L, t1_L, s1_R, v1_R, t1_R = G(s1_last)
        Keep = ''
        Lose = ''
        if alpha_bits[i] == 0:
            Keep = 'L'
            Lose = 'R'
        else:
            Keep = 'R'
            Lose = 'L'
        s0_Lose = locals()['s0_' + Lose]
        s1_Lose = locals()['s1_' + Lose]
        v1_Lose = locals()['v1_' + Lose]
        v0_Lose = locals()['v0_' + Lose]

        s0_Keep = locals()['s0_' + Keep]
        s1_Keep = locals()['s1_' + Keep]
        t0_Keep = locals()['t0_' + Keep]
        t1_Keep = locals()['t1_' + Keep]
        v1_Keep = locals()['v1_' + Keep]
        v0_Keep = locals()['v0_' + Keep]

        sCW = jnp.bitwise_xor(s0_Lose, s1_Lose)

        VCW = (-1) ** (t1_last) * (convert_G(v0_Lose) - convert_G(v1_Lose) - Valpha)

        if Lose == 'L':
            VCW = VCW + (-1) ** (t1_last) * beta

        Valpha = (
            Valpha - convert_G(v1_Keep) + convert_G(v0_Keep) + (-1) ** (t1_last) * VCW
        )
        tCW_L = jnp.bitwise_xor(
            jnp.bitwise_xor(jnp.bitwise_xor(t0_L, t1_L), alpha_bits[i]), 1
        )
        tCW_R = jnp.bitwise_xor(jnp.bitwise_xor(t0_R, t1_R), alpha_bits[i])
        CW_i = [sCW, VCW, tCW_L, tCW_R]
        CWs.append(CW_i)

        s0_last = jnp.bitwise_xor(s0_Keep, t0_last * sCW)
        s1_last = jnp.bitwise_xor(s1_Keep, t1_last * sCW)
        t0_last = jnp.bitwise_xor(t0_Keep, VCW)
        t1_last = jnp.bitwise_xor(t1_Keep, VCW)
    CW = (-1) ** (t1_last) * (convert_G(s1_last) - convert_G(s0_last) - Valpha)
    CWs.append(CW)
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
    V = 0
    t = b

    for i in range(GROUP_BIT_NUM):
        sCW, VCW, tCW_L, tCW_R = CWs[i]
        s_hat_L, v_hat_L, t_hat_L, s_hat_R, v_hat_R, t_hat_R = G(s)
        s_L = jnp.bitwise_xor(s_hat_L, t * sCW)
        t_L = jnp.bitwise_xor(t_hat_L, t * tCW_L)
        s_R = jnp.bitwise_xor(s_hat_R, t * sCW)
        t_R = jnp.bitwise_xor(t_hat_R, t * tCW_R)
        if x_bits[i] == 0:
            V = V + (-1) ** b * (convert_G(v_hat_L) + t * VCW)
            s = s_L
            t = t_L
        else:
            V = V + (-1) ** b * (convert_G(v_hat_R) + t * VCW)
            s = s_R
            t = t_R
    V = V + (-1) ** b * (convert_G(s) + t * CWs[GROUP_BIT_NUM])
    return V


if __name__ == '__main__':
    # generate key for function < 0 return -1, else 0
    k0, k1 = DCF_gen(64, 0, -1, 123)
    print(k0, k1)

    eval0 = DCF_eval(0, k0, 1000)
    eval1 = DCF_eval(1, k1, 1000)
    print(eval0, eval1, type(eval0), type(eval1))
    assert eval0 + eval1 == 0, f"eval0 + eval1 != 0,{eval0+eval1}"
