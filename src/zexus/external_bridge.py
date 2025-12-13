# External function bridge for Zexus
import hashlib

external_functions = {
    "sha256_hash": lambda data: hashlib.sha256(data.encode()).hexdigest(),
    "generate_sphincs_keypair": lambda: {
        "public_key": "sphincs_pub_placeholder", 
        "private_key": "sphincs_priv_placeholder"
    }
}

def call_external(function_name, args):
    if function_name in external_functions:
        return external_functions[function_name](*args)
    else:
        raise Exception(f"External function not found: {function_name}")
