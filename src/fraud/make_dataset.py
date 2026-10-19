from fraud.config import PROCESSED_DIR
from fraud.data import load_raw, time_split


def main():
    train, val, test = time_split(load_raw())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows, fraud rate {part['Class'].mean():.4%}")


if __name__ == "__main__":
    main()