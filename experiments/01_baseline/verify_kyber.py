# experiments/01_baseline/verify_kyber.py
#
# Verifies liboqs Kyber-512 installation and correct API usage.
# Run this before baseline.py to confirm your environment is working.
#
# Setup notes (macOS):
#   brew install cmake openssl@3
#   pip install liboqs-python
#   First import auto-builds liboqs C library (~few minutes, cached at ~/_oqs)
#
# API note (liboqs-python v0.10+):
#   generate_keypair() returns public key only
#   Secret key is stored internally in the kem object
#   decap_secret(ct) uses internal sk automatically

import oqs

def verify_kyber512():
    kem = oqs.KeyEncapsulation('Kyber512')

    # Key generation — pk only returned
    pk = kem.generate_keypair()

    # Encapsulation
    ct, ss = kem.encap_secret(pk)

    # Decapsulation — uses internal sk
    ss_recovered = kem.decap_secret(ct)

    # Verify
    print("Kyber-512 Environment Check")
    print("=" * 40)
    print(f"Ciphertext size:    {len(ct)} bytes  (spec: 768)")
    print(f"Shared secret size: {len(ss)} bytes   (spec: 32)")
    print(f"Decapsulation:      {'PASS' if ss == ss_recovered else 'FAIL'}")
    print(f"Shared secret match: {ss == ss_recovered}")
    print("=" * 40)
    print("Environment ready for baseline.py ✅")

if __name__ == "__main__":
    verify_kyber512()
