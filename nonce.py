import hashlib
from multiprocessing import Pool, cpu_count

TOKEN = "db4df190027139ba"
DIFFICULTY = 28

def leading_zero_bits(digest):
    bits = 0
    for byte in digest:
        if byte == 0:
            bits += 8
        else:
            bits += 8 - byte.bit_length()
            return bits
    return bits

def search(args):
    start, step = args
    nonce = start
    while True:
        h = hashlib.sha256(f"{TOKEN}:{nonce}".encode()).digest()
        if leading_zero_bits(h) >= DIFFICULTY:
            return nonce
        nonce += step

if __name__ == "__main__":
    workers = cpu_count()
    print(f"Mining with {workers} workers, difficulty {DIFFICULTY}...")
    with Pool(workers) as pool:
        # Each worker starts at a different offset, steps by #workers
        results = pool.imap_unordered(search, [(i, workers) for i in range(workers)])
        nonce = next(results)
        pool.terminate()
    print("NONCE:", nonce)
    # Verify
    h = hashlib.sha256(f"{TOKEN}:{nonce}".encode()).hexdigest()
    print("HASH:", h)
    print("Leading zero bits:", leading_zero_bits(bytes.fromhex(h)))
