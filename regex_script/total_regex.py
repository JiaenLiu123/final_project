from unidecode import unidecode
import scipy.stats

# FR Locale

GOOD_DECIMAL_SEPS = ["", ",", "."]
MIN_REASONABLE = 0.1

empty_sep = (
    lambda name, max_size=2, min_size=0: f"(?P<{name}>[^\d]{{{min_size},{max_size}}})"
)

adjacent_texts = [
    "total(?! (?:ht|tva))",
    "tot.",
    "ttc",
    "montant",
    "a payer",
    "montant du",
    "avec avantages",
    "apres remises",
    "carte bancaire",
    "\d+ art(?:icle(?:\(s\))?)?s?",
]
adjacent = lambda name: f'(?P<{name}>{"|".join(adjacent_texts)})'

currecy_indicators = ["€", "euros", "euro", "eur", "(?<!\w)e", "€uros", "€uro", "€ur"]
currency = lambda name: f'(?P<{name}>(?:{"|".join(currecy_indicators)}))'

integer_part = "(?P<integer_part>(?:[1-9]\d+)|\d)"
decimal_part = "(?P<decimal_part>\d\d?)"

regex_prefix = f"(?=({adjacent('adjacent')}{empty_sep('sep_prefix', 4)})?({currency('currency_prefix')}{empty_sep('sep_currency_prefix')})?(?:{integer_part}{empty_sep('sep_decimal', 2, 1)}{decimal_part})({empty_sep('sep_currency_suffix')}{currency('currency_suffix')})?)"
regex_suffix = f"(?=({currency('currency_prefix')}{empty_sep('sep_currency_prefix')})?(?:{integer_part}{empty_sep('sep_decimal', 2, 1)}{decimal_part})({empty_sep('sep_currency_suffix')}{currency('currency_suffix')})?({empty_sep('sep_suffix', 4)}{adjacent('adjacent')})?)"

MAX_REASONABLE = 250


def naive_score_total(possibility, len_matches):
    if possibility["total"] < MIN_REASONABLE:
        return 0
    return (
        round(
            160 * possibility["has_adjacent"]
            + 70 * possibility["has_currency"]
            + 50 * (possibility["sep_decimal"] in GOOD_DECIMAL_SEPS)
            + (1 / (1 + len_matches - possibility["index"])) * 30
            + (
                0
                if possibility["total"] > MAX_REASONABLE
                else scipy.stats.norm(MAX_REASONABLE, MAX_REASONABLE / 4).pdf(
                    possibility["total"]
                )
                * 70
            ),
            2,
        )
        / 300
    )
