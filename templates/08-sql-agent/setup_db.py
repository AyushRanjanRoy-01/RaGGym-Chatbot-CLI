"""
setup_db.py — Create and seed financials.db with sample tables.
Idempotent: drops and recreates tables on each run.
"""
import sqlite3

DB_PATH = "financials.db"


def setup():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop existing tables for idempotency
    cur.execute("DROP TABLE IF EXISTS quarterly_financials")
    cur.execute("DROP TABLE IF EXISTS companies")

    cur.execute("""
        CREATE TABLE companies (
            id      INTEGER PRIMARY KEY,
            name    TEXT NOT NULL,
            ticker  TEXT NOT NULL,
            sector  TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE quarterly_financials (
            id             INTEGER PRIMARY KEY,
            company_id     INTEGER REFERENCES companies(id),
            fiscal_quarter TEXT NOT NULL,   -- e.g. 'Q1 2024'
            revenue        REAL NOT NULL,   -- USD
            net_income     REAL NOT NULL,   -- USD
            ebitda         REAL NOT NULL    -- USD
        )
    """)

    # Seed companies
    companies = [
        (1, "Acme Corp",      "ACME", "Technology"),
        (2, "BrightFinance",  "BFIN", "Financial Services"),
        (3, "GreenEnergy Co", "GREC", "Energy"),
    ]
    cur.executemany("INSERT INTO companies VALUES (?,?,?,?)", companies)

    # Seed quarterly financials (revenue, net_income, ebitda in USD)
    rows = [
        # Acme Corp
        (1,  1, "Q1 2024", 4_200_000,  630_000,  840_000),
        (2,  1, "Q2 2024", 4_550_000,  682_500,  910_000),
        (3,  1, "Q3 2024", 4_800_000,  720_000,  960_000),
        (4,  1, "Q4 2024", 5_100_000,  765_000, 1_020_000),
        # BrightFinance
        (5,  2, "Q1 2024", 8_100_000, 1_620_000, 2_025_000),
        (6,  2, "Q2 2024", 8_500_000, 1_700_000, 2_125_000),
        (7,  2, "Q3 2024", 9_200_000, 1_840_000, 2_300_000),
        (8,  2, "Q4 2024", 9_750_000, 1_950_000, 2_437_500),
        # GreenEnergy Co
        (9,  3, "Q1 2024", 3_300_000,  330_000,  660_000),
        (10, 3, "Q2 2024", 3_600_000,  360_000,  720_000),
        (11, 3, "Q3 2024", 3_900_000,  390_000,  780_000),
        (12, 3, "Q4 2024", 4_100_000,  410_000,  820_000),
    ]
    cur.executemany(
        "INSERT INTO quarterly_financials VALUES (?,?,?,?,?,?)", rows
    )

    conn.commit()
    conn.close()
    print(f"Created {DB_PATH} with {len(companies)} companies and {len(rows)} quarterly records.")


if __name__ == "__main__":
    setup()
