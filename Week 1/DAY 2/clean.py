
import pandas as pd
from rapidfuzz import process, fuzz
import os


# =========================================================
# LOAD DATA
# =========================================================

def load_data(folder_path="../../Olist_Data"):
    """
    Loads all 8 Olist CSV files into a dictionary of DataFrames.
    Keeping them in a dictionary makes it easy to loop over every
    table at once for checks like nulls and duplicates.
    """
    df = {
        'customers': pd.read_csv(f'{folder_path}/olist_customers_dataset.csv'),
        'orders': pd.read_csv(f'{folder_path}/olist_orders_dataset.csv'),
        'order_items': pd.read_csv(f'{folder_path}/olist_order_items_dataset.csv'),
        'payments': pd.read_csv(f'{folder_path}/olist_order_payments_dataset.csv'),
        'reviews': pd.read_csv(f'{folder_path}/olist_order_reviews_dataset.csv'),
        'products': pd.read_csv(f'{folder_path}/olist_products_dataset.csv'),
        'sellers': pd.read_csv(f'{folder_path}/olist_sellers_dataset.csv'),
        'geolocation': pd.read_csv(f'{folder_path}/olist_geolocation_dataset.csv'),
    }
    return df


# =========================================================
# CONCERN 1: MISSING VALUES
# =========================================================

def clean_missing_values(df):
    """
    Handles missing values table by table. For every column with nulls,
    we decided WHY it's missing (MCAR, MAR, or MNAR) before deciding
    what to do about it - drop, fill in, or just flag it.
    """

    # ---------- ORDERS TABLE ----------

    # order_approved_at missing (0.16%) - probably approved right after purchase
    # type: MAR - depends on order status
    # fix: just use purchase time instead
    df['orders']['order_approved_at'] = df['orders']['order_approved_at'].fillna(
        df['orders']['order_purchase_timestamp']
    )

    # order_delivered_carrier_date missing (1.79%) - order was never shipped
    # type: MAR - tied to order status
    # fix: don't guess a date, just flag it
    df['orders']['carrier_date_missing'] = df['orders']['order_delivered_carrier_date'].isnull().astype(int)

    # order_delivered_customer_date missing (2.98%) - order was never delivered
    # type: MAR - same reason as above
    # fix: keep it empty, just flag it
    df['orders']['customer_date_missing'] = df['orders']['order_delivered_customer_date'].isnull().astype(int)

    # ---------- REVIEWS TABLE ----------

    # review_comment_title missing (88.3%) - customer just didn't write a title
    # type: MAR - depends on whether they wrote anything at all
    # fix: fill with "no_title" and flag it
    df['reviews']['title_missing'] = df['reviews']['review_comment_title'].isnull().astype(int)
    df['reviews']['review_comment_title'] = df['reviews']['review_comment_title'].fillna('no_title')

    # review_comment_message missing (58.7%) - same idea, no comment left
    # type: MAR
    # fix: fill with "no_comment" and flag it
    df['reviews']['comment_missing'] = df['reviews']['review_comment_message'].isnull().astype(int)
    df['reviews']['review_comment_message'] = df['reviews']['review_comment_message'].fillna('no_comment')

    # ---------- PRODUCTS TABLE ----------

    # product_category_name missing (1.85%) - some products just never got categorized
    # type: MNAR - could be tied to the product itself being obscure/incomplete
    # fix: fill with "unknown"
    df['products']['product_category_name'] = df['products']['product_category_name'].fillna('unknown')

    # product_name_lenght, product_description_lenght, product_photos_qty missing (same 1.85%)
    # these are missing on the same rows as category, so probably incomplete listings
    # type: MAR
    # fix: fill with median
    df['products']['product_name_lenght'] = df['products']['product_name_lenght'].fillna(
        df['products']['product_name_lenght'].median()
    )
    df['products']['product_description_lenght'] = df['products']['product_description_lenght'].fillna(
        df['products']['product_description_lenght'].median()
    )
    df['products']['product_photos_qty'] = df['products']['product_photos_qty'].fillna(
        df['products']['product_photos_qty'].median()
    )

    # weight/length/height/width missing (only 2 rows each, tiny)
    # type: MCAR - too few rows to see a pattern, probably just a random data entry gap
    # fix: fill with median, barely affects anything
    for col in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
        df['products'][col] = df['products'][col].fillna(df['products'][col].median())

    return df


# =========================================================
# CONCERN 2: DUPLICATES
# =========================================================

def clean_duplicates(df):
    """
    Checks every table for exact duplicate rows and removes them where found.
    Only geolocation had duplicates (261,831 rows) - same lat/lng recorded
    more than once for the same zip code, which adds no new information.
    """
    for name, table in df.items():
        dup_count = table.duplicated().sum()
        if dup_count > 0:
            df[name] = table.drop_duplicates()
            print(f"{name}: removed {dup_count} exact duplicate rows")

    return df


# =========================================================
# CONCERN 3: INCONSISTENT NAMING + WRONG DTYPES
# =========================================================

def clean_naming_and_dtypes(df):
    """
    Fixes messy city names (casing, whitespace, typos/accents) using
    basic cleanup plus rapidfuzz for near-duplicate matching, and fixes
    columns that were loaded with the wrong data type (dates as text,
    category-like columns as generic text).
    """

    # ---------- FIX MESSY CITY NAMES ----------

    # step 1: basic cleanup - lowercase and strip whitespace
    # this alone catches a lot of casing/spacing inconsistencies
    df['geolocation']['geolocation_city'] = df['geolocation']['geolocation_city'].str.lower().str.strip()

    # step 2: use rapidfuzz to catch the remaining variants (accents, typos)
    # that basic cleanup can't fix - e.g. "sao paulo" vs "são paulo"
    # we build a canonical list from the most common city name per zip code,
    # since a zip code should map to one consistent city name
    most_common_city = df['geolocation'].groupby('geolocation_zip_code_prefix')['geolocation_city'].agg(
        lambda x: x.mode()[0]
    )
    canonical_cities = most_common_city.unique()

    unique_cities = df['geolocation']['geolocation_city'].unique()
    replacements = {}

    for city in unique_cities:
        match, score, idx = process.extractOne(city, canonical_cities, scorer=fuzz.ratio)
        # only auto-fix names that are a close match (90+) but not identical (100)
        # score below 90 risks wrong matches (e.g. "sp" wrongly matching "sape")
        # we checked a sample of these matches by hand before trusting this cutoff
        if 90 <= score < 100:
            replacements[city] = match

    df['geolocation']['geolocation_city'] = df['geolocation']['geolocation_city'].replace(replacements)
    print(f"naming fix: cleaned {len(replacements)} inconsistent city name variants")

    # ---------- FIX DATE COLUMNS STORED AS TEXT ----------

    date_columns = {
        'orders': [
            'order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ],
        'order_items': ['shipping_limit_date'],
        'reviews': ['review_creation_date', 'review_answer_timestamp'],
    }

    for table_name, columns in date_columns.items():
        for col in columns:
            df[table_name][col] = pd.to_datetime(df[table_name][col])

    # ---------- FIX CATEGORY-LIKE COLUMNS STORED AS GENERIC TEXT ----------

    df['orders']['order_status'] = df['orders']['order_status'].astype('category')
    df['payments']['payment_type'] = df['payments']['payment_type'].astype('category')
    df['products']['product_category_name'] = df['products']['product_category_name'].astype('category')
    df['customers']['customer_state'] = df['customers']['customer_state'].astype('category')
    df['sellers']['seller_state'] = df['sellers']['seller_state'].astype('category')
    df['geolocation']['geolocation_state'] = df['geolocation']['geolocation_state'].astype('category')

    print("dtype fixes done: dates converted, category columns converted")

    return df


# =========================================================
# CONCERN 4: OUTLIERS
# =========================================================

def detect_outliers(df):
    """
    Checks price and freight_value for outliers using two methods
    (IQR and z-score), then manually inspects the flagged rows to
    decide whether to keep, cap, or remove them. Both methods will
    flag different amounts - that's expected, not a bug.
    """

    # ---------- PRICE ----------

    Q1 = df['order_items']['price'].quantile(0.25)
    Q3 = df['order_items']['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    price_outliers_iqr = df['order_items'][
        (df['order_items']['price'] < lower_bound) | (df['order_items']['price'] > upper_bound)
    ]

    mean_price = df['order_items']['price'].mean()
    std_price = df['order_items']['price'].std()
    price_zscore = (df['order_items']['price'] - mean_price) / std_price
    price_outliers_zscore = df['order_items'][price_zscore.abs() > 3]

    print(f"price: IQR flagged {len(price_outliers_iqr)} rows, z-score flagged {len(price_outliers_zscore)} rows")

    # price: kept because manual inspection showed these are real products, not errors

    # ---------- FREIGHT_VALUE ----------

    Q1_f = df['order_items']['freight_value'].quantile(0.25)
    Q3_f = df['order_items']['freight_value'].quantile(0.75)
    IQR_f = Q3_f - Q1_f
    lower_bound_f = Q1_f - 1.5 * IQR_f
    upper_bound_f = Q3_f + 1.5 * IQR_f

    freight_outliers_iqr = df['order_items'][
        (df['order_items']['freight_value'] < lower_bound_f) | (df['order_items']['freight_value'] > upper_bound_f)
    ]

    mean_freight = df['order_items']['freight_value'].mean()
    std_freight = df['order_items']['freight_value'].std()
    freight_zscore = (df['order_items']['freight_value'] - mean_freight) / std_freight
    freight_outliers_zscore = df['order_items'][freight_zscore.abs() > 3]

    print(f"freight_value: IQR flagged {len(freight_outliers_iqr)} rows, "
          f"z-score flagged {len(freight_outliers_zscore)} rows")

    # freight_value: kept because manual inspection showed high freight matches high price, not errors

    return df


# =========================================================
# BEFORE / AFTER QUALITY REPORT
# =========================================================

def get_quality_snapshot(df):
    """
    Takes a snapshot of row counts, null percentages, and duplicate
    counts for every table. Call this once before cleaning and once
    after, so the two can be compared side by side.
    """
    snapshot = {}
    for name, table in df.items():
        snapshot[name] = {
            'row_count': len(table),
            'null_pct': (table.isnull().mean() * 100).round(2).to_dict(),
            'duplicate_count': table.duplicated().sum(),
        }
    return snapshot


def print_quality_report(before, after):
    """
    Prints a simple side-by-side comparison of the before and after
    snapshots, so it's easy to see exactly what cleaning changed.
    """
    print("\n===== DATA QUALITY REPORT: BEFORE vs AFTER =====\n")

    for name in before.keys():
        print(f"--- {name} ---")
        print(f"row count:      before={before[name]['row_count']:<10} after={after[name]['row_count']}")
        print(f"duplicate rows: before={before[name]['duplicate_count']:<10} after={after[name]['duplicate_count']}")

        print("null % by column (before -> after):")
        all_cols = set(before[name]['null_pct'].keys()) | set(after[name]['null_pct'].keys())
        for col in sorted(all_cols):
            b = before[name]['null_pct'].get(col, 0)
            a = after[name]['null_pct'].get(col, 0)
            if b > 0 or a > 0:
                print(f"  {col}: {b}% -> {a}%")
        print()


# =========================================================
# RUN EVERYTHING
# =========================================================

if __name__ == "__main__":
    df = load_data("../../Olist_Data")

    before_snapshot = get_quality_snapshot(df)

    df = clean_missing_values(df)
    df = clean_duplicates(df)
    df = clean_naming_and_dtypes(df)
    df = clean_duplicates(df)
    df = detect_outliers(df)

    after_snapshot = get_quality_snapshot(df)

    print_quality_report(before_snapshot, after_snapshot)


    os.makedirs("../../cleaned_data", exist_ok=True)
    for name, table in df.items():
        table.to_csv(f"../../cleaned_data/{name}.csv", index=False)
    print("saved cleaned tables to ../../cleaned_data/ folder")