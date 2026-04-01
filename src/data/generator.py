"""Synthetic data generator for tabular column classification.

Generates realistic column samples for each of the 8 column types:
email, phone, price, id, date, name, address, categorical.

Each sample contains:
- header: column header text
- values: list of sample cell values
- stats: statistical features (n_unique, entropy, null_ratio, mean_length)
- patterns: pattern-based features (is_email, is_phone, is_numeric, is_date, has_at, has_dot, is_text, has_space, has_digit)
- label: integer class label
"""

import math
import random
import re
import string
from typing import Any



# Template pools for generating realistic column data
_HEADERS = {
    "email": [
        "email", "e-mail", "email_address", "mail", "contact_email",
        "user_email", "Email", "EMAIL", "courriel", "email_pro",
    ],
    "phone": [
        "phone", "telephone", "phone_number", "tel", "mobile",
        "cell", "contact_phone", "Phone", "TEL", "fax",
    ],
    "price": [
        "price", "amount", "cost", "total", "unit_price",
        "prix", "montant", "Price", "PRICE", "fee",
    ],
    "id": [
        "id", "ID", "identifier", "code", "ref",
        "reference", "num", "number", "Id", "uid",
    ],
    "date": [
        "date", "created_at", "updated_at", "timestamp", "birth_date",
        "start_date", "end_date", "Date", "DATE", "order_date",
    ],
    "name": [
        "name", "full_name", "first_name", "last_name", "customer_name",
        "client_name", "nom", "prenom", "Name", "USERNAME",
    ],
    "address": [
        "address", "street", "city", "location", "addr",
        "street_address", "adresse", "Address", "ADDRESS", "residence",
    ],
    "categorical": [
        "status", "category", "type", "level", "gender",
        "country", "department", "grade", "classe", "group",
    ],
    "account_number": [
        "compte", "account", "iban", "n° de compte", "numéro de compte",
        "numéro abrégé du compte", "N° cpte", "account_number", "Compte DO",
        "cpt_iban", "numéro compte",
    ],
    "description": [
        "motif", "libellé", "description", "commentaires", "information",
        "intitulé", "libelle", "Motif de paiement", "observations", "note",
    ],
    "quantity": [
        "quantite", "volume", "nombre", "vol", "qty", "count",
        "nb", "quantity", "nombre d'opérations", "Nombre de mouvements",
    ],
}

_FIRST_NAMES = [
    "John", "Anna", "Pierre", "Marie", "James", "Sophie", "Carlos", "Emma",
    "Ahmed", "Yuki", "Chen", "Fatima", "Luca", "Olga", "Raj", "Ingrid",
]
_LAST_NAMES = [
    "Smith", "Dupont", "Garcia", "Mueller", "Tanaka", "Wang", "Ali", "Jensen",
    "Rossi", "Petrov", "Kim", "Silva", "Martin", "Brown", "Leroy", "Patel",
]
_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com", "mail.fr"]
_STREETS = [
    "Main St", "Oak Ave", "Rue de Paris", "Elm Street", "Broadway",
    "5th Avenue", "Baker Street", "Maple Drive", "Sunset Blvd", "Park Lane",
]
_CITIES = [
    "New York", "Paris", "London", "Tokyo", "Berlin",
    "Sydney", "Toronto", "Mumbai", "São Paulo", "Cairo",
]
_CATEGORIES = {
    "status": ["active", "inactive", "pending", "suspended"],
    "gender": ["male", "female", "other", "prefer_not_to_say"],
    "level": ["junior", "mid", "senior", "lead", "director"],
    "country": ["US", "FR", "DE", "JP", "BR", "IN", "UK", "CA"],
    "type": ["A", "B", "C", "D"],
}


def _generate_email() -> str:
    first = random.choice(_FIRST_NAMES).lower()
    last = random.choice(_LAST_NAMES).lower()
    domain = random.choice(_DOMAINS)
    sep = random.choice([".", "_", ""])
    num = random.choice(["", str(random.randint(1, 99))])
    return f"{first}{sep}{last}{num}@{domain}"


def _generate_phone() -> str:
    formats = [
        lambda: f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        lambda: f"+33 {random.randint(1, 9)} {random.randint(10, 99)} {random.randint(10, 99)} {random.randint(10, 99)} {random.randint(10, 99)}",
        lambda: f"({random.randint(200, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
        lambda: f"{random.randint(100, 999)}.{random.randint(100, 999)}.{random.randint(1000, 9999)}",
    ]
    return random.choice(formats)()


def _generate_price() -> str:
    amount = round(random.uniform(0.5, 9999.99), 2)
    formats = [
        lambda a: f"${a:.2f}",
        lambda a: f"{a:.2f}€",
        lambda a: f"{a:.2f}",
        lambda a: f"${a:,.2f}",
    ]
    return random.choice(formats)(amount)


def _generate_id() -> str:
    formats = [
        lambda: str(random.randint(1000, 999999)),
        lambda: f"ID-{random.randint(1, 99999):05d}",
        lambda: "".join(random.choices(string.ascii_uppercase + string.digits, k=8)),
        lambda: f"REF-{''.join(random.choices(string.ascii_uppercase, k=3))}-{random.randint(100, 999)}",
    ]
    return random.choice(formats)()


def _generate_date() -> str:
    year = random.randint(1990, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    formats = [
        lambda y, m, d: f"{y}-{m:02d}-{d:02d}",
        lambda y, m, d: f"{d:02d}/{m:02d}/{y}",
        lambda y, m, d: f"{m:02d}-{d:02d}-{y}",
        lambda y, m, d: f"{y}/{m:02d}/{d:02d}",
    ]
    return random.choice(formats)(year, month, day)


def _generate_name() -> str:
    first = random.choice(_FIRST_NAMES)
    last = random.choice(_LAST_NAMES)
    fmt = random.choice(["full", "first", "last", "last_first"])
    if fmt == "full":
        return f"{first} {last}"
    elif fmt == "first":
        return first
    elif fmt == "last":
        return last
    else:
        return f"{last}, {first}"


def _generate_address() -> str:
    num = random.randint(1, 9999)
    street = random.choice(_STREETS)
    city = random.choice(_CITIES)
    fmt = random.choice(["full", "street", "city_street"])
    if fmt == "full":
        return f"{num} {street}, {city}"
    elif fmt == "street":
        return f"{num} {street}"
    else:
        return f"{street}, {city}"
    


def _generate_account_number() -> str:
    formats = [
        # IBAN-like
        lambda: f"FR{random.randint(10,99)}{random.randint(10000,99999)}{''.join(random.choices(string.digits, k=20))}",
        # abbreviated numeric account
        lambda: str(random.randint(10000000, 999999999)),
        # alphanumeric account ref
        lambda: f"{''.join(random.choices(string.ascii_uppercase + string.digits, k=4))}{random.randint(10000, 99999)}",
        # zero-padded account
        lambda: f"0000{random.choice(string.ascii_uppercase)}{random.randint(100000, 999999)}",
    ]
    return random.choice(formats)()


def _generate_description() -> str:
    prefixes = [
        "VIREMENT EN FAVEUR DE", "REGLEMENT FACTURE", "SALAIRES ET REMUNERATIONS",
        "LOYER TRIMESTRE", "PAIEMENT REFERENCE", "REMBOURSEMENT", "HONORAIRES",
        "ANCIEN ID EVCLI =", "OPERATION INTERBANCAIRE", "MONTANT D ORIGINE",
        "PRELEVEMENT SEPA", "NOTE DE FRAIS", "COMMISSION SUR", "FRAIS DE DOSSIER",
        "RAPPORT MENSUEL", "DETAIL OPERATION", "OBSERVATION CLIENT",
    ]
    suffixes = [
        f"{random.randint(1, 9999999)}", f"REF {random.randint(1000, 9999999)}",
        f"DU {random.randint(1, 28):02d}/0{random.randint(1, 9)}/{random.randint(2020, 2025)}",
        "", f"- {random.choice(['EUR', 'USD', 'GBP'])} {random.randint(100, 99999)}",
    ]
    return f"{random.choice(prefixes)} {random.choice(suffixes)}".strip()


def _generate_quantity() -> str:
    formats = [
        # plain integer count
        lambda: str(random.randint(0, 9999999)),
        # large volume with spaces (French number formatting)
        lambda: f"{random.randint(1, 999):,}".replace(",", " ") + f" {random.randint(0, 999):03d}",
        # small count
        lambda: str(random.randint(0, 999)),
        # zero (frequent in your data)
        lambda: "0",
    ]
    return random.choice(formats)()



def _generate_categorical() -> str:
    cat_type = random.choice(list(_CATEGORIES.keys()))
    return random.choice(_CATEGORIES[cat_type])


_GENERATORS = {
    "email": _generate_email,
    "phone": _generate_phone,
    "price": _generate_price,
    "id": _generate_id,
    "date": _generate_date,
    "name": _generate_name,
    "address": _generate_address,
    "categorical": _generate_categorical,
    "account_number": _generate_account_number,
    "description":    _generate_description,
    "quantity":       _generate_quantity,
}


def _compute_entropy(values: list[str]) -> float:
    """Compute Shannon entropy of value distribution."""
    from collections import Counter
    counts = Counter(values)
    total = len(values)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _compute_stats(values: list[str]) -> dict[str, float]:
    """Compute statistical features for a column."""
    non_null = [v for v in values if v and v.strip()]
    n_unique = len(set(non_null))
    entropy = _compute_entropy(non_null)
    null_ratio = 1.0 - len(non_null) / max(len(values), 1)
    mean_length = sum(len(v) for v in non_null) / max(len(non_null), 1)
    return {
        "n_unique": float(n_unique),
        "entropy": entropy,
        "null_ratio": null_ratio,
        "mean_length": mean_length,
    }


def _compute_patterns(values: list[str]) -> dict[str, float]:
    """Compute pattern-based features for a column."""
    non_null = [v for v in values if v and v.strip()]
    n = max(len(non_null), 1)

    email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone_pattern = re.compile(r"^[\+\(\d][\d\s\-\.\(\)]{6,}$")
    date_pattern = re.compile(r"^\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}$")

    is_email = sum(1 for v in non_null if email_pattern.match(v)) / n
    is_phone = sum(1 for v in non_null if phone_pattern.match(v)) / n
    is_numeric = sum(1 for v in non_null if v.replace(".", "").replace(",", "").replace("$", "").replace("€", "").replace("-", "").strip().isdigit()) / n
    is_date = sum(1 for v in non_null if date_pattern.match(v)) / n
    has_at = sum(1 for v in non_null if "@" in v) / n
    has_dot = sum(1 for v in non_null if "." in v) / n

    # --- new features ---
    # is_text: purely alphabetic (letters + spaces + hyphens), no digits or special chars
    is_text    = sum(1 for v in non_null if v.replace(" ", "").replace("-", "").replace(",", "").isalpha()) / n
    # has_space: value contains at least one space (full names, addresses)
    has_space  = sum(1 for v in non_null if " " in v) / n
    # has_digit: value contains at least one digit (ids, prices, phones, dates)
    has_digit  = sum(1 for v in non_null if any(c.isdigit() for c in v)) / n

    return {
        "is_email": is_email,
        "is_phone": is_phone,
        "is_numeric": is_numeric,
        "is_date": is_date,
        "is_text": is_text,  
        "has_at": has_at,
        "has_dot": has_dot,
        "has_space":  has_space,  # new
        "has_digit":  has_digit, 
    }


def generate_column_sample(
    label: str,
    num_values: int = 10,
    null_probability: float = 0.05,
) -> dict[str, Any]:
    """Generate a single synthetic column sample.

    Args:
        label: Column type (e.g., 'email', 'phone')
        num_values: Number of cell values to generate
        null_probability: Probability of inserting a null value

    Returns:
        Dictionary with header, values, stats, patterns, and label
    """
    header = random.choice(_HEADERS[label])
    generator = _GENERATORS[label]

    values = []
    for _ in range(num_values):
        if random.random() < null_probability:
            values.append("")
        else:
            values.append(generator())

    stats = _compute_stats(values)
    patterns = _compute_patterns(values)

    return {
        "header": header,
        "values": values,
        "stats": stats,
        "patterns": patterns,
        "label": label,
    }


def generate_dataset(
    num_samples_per_class: int = 500,
    class_labels: list[str] | None = None,
    num_values: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a full synthetic dataset for column classification.

    Args:
        num_samples_per_class: Number of samples per class
        class_labels: List of class labels to generate
        num_values: Number of values per column sample
        seed: Random seed for reproducibility

    Returns:
        List of column sample dictionaries
    """
    random.seed(seed)

    if class_labels is None:
        class_labels = list(_HEADERS.keys())

    dataset = []
    for label in class_labels:
        for _ in range(num_samples_per_class):
            sample = generate_column_sample(label, num_values=num_values)
            dataset.append(sample)

    random.shuffle(dataset)
    return dataset
