import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
from utils.constants import KALSHI_API_KEY



def load_private_key() -> rsa.RSAPrivateKey:
    private_key = serialization.load_pem_private_key(
        KALSHI_API_KEY.encode('utf-8'),
        password=None,
        backend=default_backend()
    )
    if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("Loaded key is not an RSA private key")
    return private_key

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    except InvalidSignature as e:
        print("Failed to sign message.")
        raise ValueError("RSA sign PSS failed") from e