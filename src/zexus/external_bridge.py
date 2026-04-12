# External function bridge for Zexus
import hashlib
import secrets


def _generate_sphincs_keypair():
    """Generate a placeholder SPHINCS+ keypair.

    NOTE: These are *runtime-generated random placeholders*, not real
    post-quantum keys.  Replace with an actual PQC library (e.g.
    ``oqs-python``) when moving to production.
    """
    return {
        "public_key": secrets.token_hex(32),
        "private_key": secrets.token_hex(64),
    }


external_functions = {
    "sha256_hash": lambda data: hashlib.sha256(data.encode()).hexdigest(),
    "generate_sphincs_keypair": _generate_sphincs_keypair,
}

def call_external(function_name, args):
    if function_name in external_functions:
        return external_functions[function_name](*args)
    else:
        raise Exception(f"External function not found: {function_name}")
