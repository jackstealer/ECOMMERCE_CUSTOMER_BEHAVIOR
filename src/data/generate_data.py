"""
src/data/generate_data.py
─────────────────────────────────────────────────────────────
Generates synthetic e-commerce datasets for the ML project.

Tables created (saved to data/raw/):
  users.csv          →  5,000 rows   (user profiles)
  products.csv       →    500 rows   (product catalog)
  sessions.csv       → 20,000 rows   (web sessions)
  browse_events.csv  → ~70-90k rows  (click / view / cart events)
  orders.csv         →  8,000 rows   (purchase records)
  order_items.csv    → ~20,000 rows  (line items per order)

Usage:
  python src/data/generate_data.py
─────────────────────────────────────────────────────────────
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_US")
Faker.seed(SEED)

# ── Config ─────────────────────────────────────────────────
START_DATE = datetime(2024, 6, 1)
END_DATE   = datetime(2025, 5, 31)
N_USERS    = 5_000
N_PRODUCTS =   500
N_SESSIONS = 20_000
N_ORDERS   =  8_000
OUTPUT_DIR = "data/raw"


# ── Helpers ────────────────────────────────────────────────

def rand_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta_secs = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, delta_secs))


# ── Table generators ───────────────────────────────────────

def generate_users(n: int = N_USERS) -> pd.DataFrame:
    """
    User profiles.

    Columns:
        user_id, name, email, age, gender, city, state,
        device_type, membership, signup_date
    """
    print(f"  Generating {n:,} users ...")

    genders    = np.random.choice(["Male", "Female", "Other"], n, p=[0.47, 0.47, 0.06])
    ages       = np.clip(np.random.normal(33, 10, n).astype(int), 18, 70)
    devices    = np.random.choice(["mobile", "desktop", "tablet"], n, p=[0.55, 0.35, 0.10])
    membership = np.random.choice(["free", "silver", "gold", "platinum"], n,
                                   p=[0.55, 0.25, 0.15, 0.05])

    # Older users joined earlier
    signup_dates = [
        rand_date(START_DATE - timedelta(days=int(age * 10)), END_DATE).strftime("%Y-%m-%d")
        for age in ages
    ]

    return pd.DataFrame({
        "user_id":     [f"U{str(i).zfill(5)}" for i in range(1, n + 1)],
        "name":        [fake.name()         for _ in range(n)],
        "email":       [fake.email()        for _ in range(n)],
        "age":         ages,
        "gender":      genders,
        "city":        [fake.city()         for _ in range(n)],
        "state":       [fake.state_abbr()   for _ in range(n)],
        "device_type": devices,
        "membership":  membership,
        "signup_date": signup_dates,
    })


def generate_products(n: int = N_PRODUCTS) -> pd.DataFrame:
    """
    Product catalog.

    Columns:
        product_id, product_name, category, subcategory,
        price, rating, review_count, stock_qty, is_featured
    """
    print(f"  Generating {n:,} products ...")

    # Category → subcategory → typical price range
    catalog = {
        "Electronics":   {"subs": ["Smartphones", "Laptops", "Headphones", "Cameras", "Tablets"],
                          "price": (20, 2000)},
        "Clothing":      {"subs": ["Shirts", "Pants", "Shoes", "Jackets", "Accessories"],
                          "price": (10, 300)},
        "Home & Garden": {"subs": ["Furniture", "Kitchen", "Bedding", "Decor", "Tools"],
                          "price": (15, 1500)},
        "Sports":        {"subs": ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Cycling"],
                          "price": (10, 500)},
        "Books":         {"subs": ["Fiction", "Non-Fiction", "Academic", "Children", "Comics"],
                          "price": (5, 80)},
        "Beauty":        {"subs": ["Skincare", "Makeup", "Hair Care", "Fragrance", "Wellness"],
                          "price": (5, 200)},
    }

    rows = []
    for i in range(1, n + 1):
        cat_name = random.choice(list(catalog.keys()))
        cat      = catalog[cat_name]
        sub      = random.choice(cat["subs"])
        lo, hi   = cat["price"]
        price    = round(random.uniform(lo, hi), 2)
        rows.append({
            "product_id":   f"P{str(i).zfill(5)}",
            "product_name": f"{sub} — Model {i:03d}",
            "category":     cat_name,
            "subcategory":  sub,
            "price":        price,
            "rating":       round(random.uniform(1.5, 5.0), 1),
            "review_count": int(np.random.exponential(200)),   # long-tail reviews
            "stock_qty":    random.randint(0, 500),
            "is_featured":  random.random() < 0.12,            # ~12% featured
        })

    return pd.DataFrame(rows)


def generate_sessions(users_df: pd.DataFrame, n: int = N_SESSIONS) -> pd.DataFrame:
    """
    Web sessions — one row per user visit.

    Columns:
        session_id, user_id, session_start, session_end,
        duration_secs, device, referral_source,
        pages_visited, bounced
    """
    print(f"  Generating {n:,} sessions ...")

    user_ids = users_df["user_id"].tolist()

    # Active users visit more often — sample with replacement, weight by activity
    activity_weights = np.random.dirichlet(np.ones(len(user_ids)) * 2)

    rows = []
    for i in range(1, n + 1):
        uid      = np.random.choice(user_ids, p=activity_weights)
        start    = rand_date(START_DATE, END_DATE)
        duration = int(np.random.exponential(600))          # seconds, avg 10 min
        duration = max(10, min(duration, 7200))             # clamp 10s – 2h
        end      = start + timedelta(seconds=duration)

        rows.append({
            "session_id":      f"S{str(i).zfill(6)}",
            "user_id":         uid,
            "session_start":   start.strftime("%Y-%m-%d %H:%M:%S"),
            "session_end":     end.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_secs":   duration,
            "device":          random.choice(["mobile", "desktop", "tablet"]),
            "referral_source": random.choice(
                ["organic_search", "paid_search", "social_media", "email", "direct", "referral"],
                # p=[0.30, 0.20, 0.20, 0.15, 0.10, 0.05]  # realistic channel mix — uncomment to weight
            ),
            "pages_visited":   max(1, int(np.random.exponential(5))),
            "bounced":         random.random() < 0.38,     # 38% bounce rate
        })

    return pd.DataFrame(rows)


def generate_browse_events(sessions_df: pd.DataFrame,
                            products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Granular click / view / cart events within sessions.

    Columns:
        event_id, session_id, user_id, product_id,
        event_type, time_spent_secs, event_sequence
    """
    print(f"  Generating browse events for {len(sessions_df):,} sessions ...")

    product_ids = products_df["product_id"].tolist()

    # Popular products follow a power-law distribution
    pop_weights = np.random.dirichlet(np.ones(len(product_ids)) * 0.3)

    event_types  = ["view", "click", "add_to_cart", "remove_from_cart", "add_to_wishlist"]
    event_probs  = [0.50,   0.25,    0.14,          0.06,               0.05]

    rows = []
    for _, session in sessions_df.iterrows():
        # Bounced sessions → 1 event only; others → up to 12
        n_events = 1 if session["bounced"] else random.randint(1, 12)

        for seq in range(1, n_events + 1):
            rows.append({
                "event_id":        f"E{str(len(rows) + 1).zfill(7)}",
                "session_id":      session["session_id"],
                "user_id":         session["user_id"],
                "product_id":      np.random.choice(product_ids, p=pop_weights),
                "event_type":      np.random.choice(event_types, p=event_probs),
                "time_spent_secs": random.randint(2, 300),
                "event_sequence":  seq,
            })

    return pd.DataFrame(rows)


def generate_orders(users_df: pd.DataFrame, n: int = N_ORDERS) -> pd.DataFrame:
    """
    Purchase records — only ~40% of users ever buy.

    Columns:
        order_id, user_id, order_date, total_amount,
        status, payment_method, discount_pct
    """
    print(f"  Generating {n:,} orders ...")

    # Buyers are a subset; gold/platinum members buy more
    premium = users_df[users_df["membership"].isin(["gold", "platinum"])]["user_id"].tolist()
    regular = users_df[users_df["membership"].isin(["free", "silver"])]["user_id"].tolist()

    # 70% of orders from premium (smaller) group — they are your best customers
    buyer_pool = random.choices(premium, k=int(n * 0.70)) + \
                 random.choices(regular, k=int(n * 0.30))

    statuses = ["delivered", "shipped", "processing", "cancelled", "returned"]
    s_probs  = [0.65,        0.15,      0.10,         0.06,         0.04]

    rows = []
    for i in range(1, n + 1):
        rows.append({
            "order_id":        f"O{str(i).zfill(6)}",
            "user_id":         buyer_pool[i - 1],
            "order_date":      rand_date(START_DATE, END_DATE).strftime("%Y-%m-%d %H:%M:%S"),
            "total_amount":    round(random.uniform(5.0, 1500.0), 2),
            "status":          np.random.choice(statuses, p=s_probs),
            "payment_method":  random.choice(["credit_card", "debit_card", "paypal", "upi", "wallet"]),
            "discount_pct":    random.choice([0, 0, 0, 5, 10, 15, 20, 25]),  # 0 most common
        })

    return pd.DataFrame(rows)


def generate_order_items(orders_df: pd.DataFrame,
                          products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Line items for every order (1–5 products each).

    Columns:
        order_item_id, order_id, product_id,
        quantity, unit_price
    """
    print(f"  Generating order items for {len(orders_df):,} orders ...")

    product_ids = products_df["product_id"].tolist()
    price_map   = dict(zip(products_df["product_id"], products_df["price"]))

    rows = []
    for _, order in orders_df.iterrows():
        n_items  = random.randint(1, 5)
        products = random.sample(product_ids, min(n_items, len(product_ids)))

        for pid in products:
            rows.append({
                "order_item_id": f"OI{str(len(rows) + 1).zfill(7)}",
                "order_id":      order["order_id"],
                "product_id":    pid,
                "quantity":      random.randint(1, 4),
                "unit_price":    price_map[pid],
            })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print(" E-Commerce Synthetic Data Generator")
    print(f" Output → {OUTPUT_DIR}/")
    print(f"{'='*55}\n")

    # Generate
    users    = generate_users()
    products = generate_products()
    sessions = generate_sessions(users)
    events   = generate_browse_events(sessions, products)
    orders   = generate_orders(users)
    items    = generate_order_items(orders, products)

    # Save
    datasets = {
        "users":         users,
        "products":      products,
        "sessions":      sessions,
        "browse_events": events,
        "orders":        orders,
        "order_items":   items,
    }

    print(f"\n{'─'*55}")
    print(f"  {'Table':<20} {'Rows':>8}   {'Cols':>5}   File")
    print(f"{'─'*55}")

    for name, df in datasets.items():
        path = f"{OUTPUT_DIR}/{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name:<20} {len(df):>8,}   {df.shape[1]:>5}   {path}")

    print(f"{'─'*55}")
    total_rows = sum(len(df) for df in datasets.values())
    print(f"  {'TOTAL':<20} {total_rows:>8,}")
    print(f"\n  All files saved to {OUTPUT_DIR}/")
    print("  Run the Phase 1 notebook next.\n")


if __name__ == "__main__":
    main()