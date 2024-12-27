from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

import secretflow as sf
from secretflow.device import Device, DeviceObject

jax.config.update("jax_enable_x64", True)


@dataclass
class Devices:
    # may extend to hold list of devices.
    edge_device_i: Device
    edge_device_j: Device
    server_device: Device

    edge_tee_i: Device
    edge_tee_j: Device
    server_tee: Device


@dataclass
class Handles:
    server_i_s: DeviceObject = None
    server_j_s: DeviceObject = None
    i_s: DeviceObject = None
    j_s: DeviceObject = None
    j_i: DeviceObject = None
    i_j: DeviceObject = None


@dataclass
class Params:
    fxp: int
    fxp_type: any
    kappa: int
    # k is the ring size 2^k. usually take k = 32, 64. like size of int
    k: int
    # m is the size of array in cos computation
    m: int


def bytes_to_jax_random_key(byte_key):
    seed = int.from_bytes(byte_key[:4], 'big')

    # Create a JAX random key with this seed
    jax_key = jax.random.PRNGKey(seed)

    return jax_key


# Function to encrypt a jnp array using AES-GCM
def encrypt_jnp_array_gcm(jnp_array, key) -> Tuple[bytes, bytes, bytes]:

    # Convert numpy array to bytes
    array_bytes = jnp_array.tobytes()

    # Create AES cipher in GCM mode
    cipher = AES.new(key, AES.MODE_GCM)

    # Encrypt data
    ciphertext, tag = cipher.encrypt_and_digest(array_bytes)

    # Return the ciphertext, tag, and nonce
    return ciphertext, tag, cipher.nonce


# Function to decrypt a jnp array using AES-GCM
def decrypt_to_jnp_array_gcm(ciphertext, tag, nonce, key, dtype, shape):
    # Create AES cipher in GCM mode with the same parameters
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

    # Decrypt data
    decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)

    # Convert bytes back to numpy array
    decrypted_jnp_array = jnp.frombuffer(decrypted_data, dtype=dtype).reshape(shape)

    return decrypted_jnp_array


def devices_enable_x64(devices: Devices):
    set_ups = [
        devices.edge_device_i(lambda: jax.config.update("jax_enable_x64", True))(),
        devices.edge_device_j(lambda: jax.config.update("jax_enable_x64", True))(),
        devices.server_device(lambda: jax.config.update("jax_enable_x64", True))(),
        devices.edge_tee_i(lambda: jax.config.update("jax_enable_x64", True))(),
        devices.edge_tee_j(lambda: jax.config.update("jax_enable_x64", True))(),
        devices.server_tee(lambda: jax.config.update("jax_enable_x64", True))(),
    ]
    sf.wait(set_ups)


def gen_handles(devices: Devices, params: Params) -> Handles:
    handles = Handles()
    # Simulate handles
    handles.i_j = devices.edge_tee_i(lambda x: get_random_bytes(x))(params.kappa)
    handles.i_s = devices.edge_tee_i(lambda x: get_random_bytes(x))(params.kappa)

    # note that the establishment is not simplified.
    handles.j_i = handles.i_j.to(devices.edge_tee_j)
    handles.j_s = devices.edge_tee_j(lambda x: get_random_bytes(x))(params.kappa)

    handles.server_i_s = handles.i_s.to(devices.server_tee)
    handles.server_j_s = handles.j_s.to(devices.server_tee)
    return handles


def simulate_data_u(
    devices: Devices, u_low: float, u_high: float, m: int
) -> Tuple[DeviceObject, DeviceObject]:
    # P_i holds u_i
    u_i = devices.edge_device_i(lambda x: x)(
        jnp.array(np.random.uniform(u_low, u_high, (m,)))
    )

    # P_j holds u_j
    u_j = devices.edge_device_j(lambda x: x)(
        jnp.array(np.random.uniform(u_low, u_high, (m,)))
    )
    return u_i, u_j


def corr(k, m, dev1, key1, dev2, key2, return_zero_sharing=False):
    """Correlation function

    Args:
        k (int): ring size will be 2^k. Support k = 64 or 128 for now
        m (int): size of array to be correlated
        dev1 (Device): device 1
        key1 (Key): key for device 1
        dev2 (Device): device 2
        key2 (Key): key for device 2, key2 must be the same as key1 yet hold by different device

    Note that key splitting functionality is not implemented here, so the random key is not updated
    however, key updating must be implemented in real production code.
    See https://jax.readthedocs.io/en/latest/key-concepts.html#key-concepts-prngs.
    """
    assert k == 64, "Only support k = 64 for now"
    dtype = jnp.uint64
    corr_dev1 = dev1(
        lambda key, shape, dtype: dtype(
            jax.random.bits(bytes_to_jax_random_key(key), shape)
        )
    )(key1, (m,), dtype)
    if not return_zero_sharing:
        corr_dev2 = dev2(
            lambda key, shape, dtype: dtype(
                jax.random.bits(bytes_to_jax_random_key(key), shape)
            )
        )(key2, (m,), dtype)
    else:
        corr_dev2 = dev2(
            lambda key, shape, dtype: dtype(
                -jax.random.bits(bytes_to_jax_random_key(key), shape)
            )
        )(key2, (m,), dtype)
    return corr_dev1, corr_dev2


# Example usage
if __name__ == "__main__":
    # Generate a random key for AES-256 (32 bytes)
    key = get_random_bytes(32)

    u_low = 0.0
    u_high = 2.0
    m = 100

    # Create a jnp array
    original_jnp_array = jnp.uint64(
        jnp.array(np.random.uniform(u_low, u_high, (m,)), dtype=jnp.float64)
    )

    # Encrypt the jnp array using AES-GCM
    ciphertext, tag, nonce = encrypt_jnp_array_gcm(original_jnp_array, key)

    # Decrypt to a jnp array
    decrypted_jnp_array = decrypt_to_jnp_array_gcm(
        ciphertext, tag, nonce, key, dtype=jnp.uint64, shape=original_jnp_array.shape
    )

    # Check if the original and decrypted arrays are the same
    print("Original JAX Array:")
    print(original_jnp_array)
    print(original_jnp_array.dtype)
    print("\nDecrypted JAX Array:")
    print(decrypted_jnp_array)
    print(decrypted_jnp_array.dtype)
    print(
        "\nArrays are equal:", jnp.array_equal(original_jnp_array, decrypted_jnp_array)
    )
