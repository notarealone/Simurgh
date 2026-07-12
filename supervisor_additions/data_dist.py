import json

def approx_token_count(text: str) -> float:
    """
    Approximate token count from character count.
    Assumption: 1000 characters ≈ 500 tokens → tokens = chars * 0.5
    """
    return len(text) * 0.5

def main(file_path: str):
    query_lengths = []
    doc_lengths = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            query = data.get('query', '')
            query_lengths.append(approx_token_count(query))

            for doc in data.get('docs', []):
                text = doc.get('text', '')
                doc_lengths.append(approx_token_count(text))

    if query_lengths:
        print("=== Query length statistics (approx tokens) ===")
        print(f"  Min  : {min(query_lengths):.2f}")
        print(f"  Max  : {max(query_lengths):.2f}")
        print(f"  Mean : {sum(query_lengths) / len(query_lengths):.2f}")
    else:
        print("No queries found.")

    if doc_lengths:
        print("\n=== Document length statistics (approx tokens) ===")
        print(f"  Min  : {min(doc_lengths):.2f}")
        print(f"  Max  : {max(doc_lengths):.2f}")
        print(f"  Mean : {sum(doc_lengths) / len(doc_lengths):.2f}")
    else:
        print("No documents found.")

if __name__ == "__main__":
    main("train.txt")