import re
from unidecode import unidecode
from regex_script.total_regex import naive_score_total, regex_suffix, regex_prefix
from regex_script.date_regex import (
    naive_date_regex,
    naive_score_date,
    regex_date_suffix_time,
    regex_date_prefix_time,
    regex_date_suffix_time_en_format,
    regex_date_prefix_time_en_format,
    parse_datetime,
)


def test_regex(text: str, verbose=False):
    text = unidecode(text).lower()

    date_matches = (
        [(match, i) for i, match in enumerate(re.finditer(naive_date_regex, text))]
        + [
            (match, i)
            for i, match in enumerate(re.finditer(regex_date_prefix_time, text))
        ]
        + [
            (match, i)
            for i, match in enumerate(re.finditer(regex_date_suffix_time, text))
        ]
        + [
            (match, i)
            for i, match in enumerate(
                re.finditer(regex_date_prefix_time_en_format, text)
            )
        ]
        + [
            (match, i)
            for i, match in enumerate(
                re.finditer(regex_date_suffix_time_en_format, text)
            )
        ]
    )

    total_matches = [
        (match, i) for i, match in enumerate(re.finditer(regex_prefix, text))
    ] + [(match, i) for i, match in enumerate(re.finditer(regex_suffix, text))]

    possible_dates = []
    for match, i in date_matches:
        try:
            (
                parsed_date,
                is_year_two_digit,
                is_day_one_digit,
                is_month_one_digit,
                is_month_name,
                has_time,
            ) = parse_datetime(match)

            try:
                sep_day = match.group("sep_day").strip()
            except:
                sep_day = match.group("sep_day_inner")

            try:
                sep_month = match.group("sep_month").strip()
            except:
                sep_month = match.group("sep_month_inner")

            sep_hour = (
                match.group("sep_hour").strip() if match.group("sep_hour") else None
            )
            sep_minute = (
                match.group("sep_minute").strip() if match.group("sep_minute") else None
            )

            try:
                sep_date = match.group("sep_date_inner").strip()
            except:
                try:
                    sep_date = match.group("sep_date_outer").strip()
                except:
                    sep_date = None

            try:
                sep_time = match.group("sep_time_inner").strip()
            except:
                try:
                    sep_time = match.group("sep_time_outer").strip()
                except:
                    sep_time = None

            possible_dates.append(
                (
                    naive_score_date(
                        {
                            "date": parsed_date,
                            "is_day_one_digit": is_day_one_digit,
                            "is_month_one_digit": is_month_one_digit,
                            "is_year_two_digit": is_year_two_digit,
                            "sep_day": sep_day,
                            "sep_month": sep_month,
                            "sep_hour": sep_hour,
                            "sep_minute": sep_minute,
                            "sep_time": sep_time,
                            "sep_date": sep_date,
                            "is_month_name": is_month_name,
                            "has_time": has_time,
                            "index": i,
                        },
                        len(date_matches),
                    ),
                    parsed_date,
                )
            )

        except Exception as e:
            print(e)
            pass  # Impossible date found

    possible_total = []
    get_decimal_part = lambda match: int(match) * (10 if len(match) == 1 else 1) / 100

    for match, i in total_matches:
        total_amount = round(
            int(match.group("integer_part"))
            + get_decimal_part(match.group("decimal_part")),
            2,
        )
        sep_decimal = match.group("sep_decimal").strip()

        has_adjacent = match.group("adjacent") is not None
        has_currency = (match.group("currency_prefix") is not None) ^ (
            match.group("currency_suffix") is not None
        )

        possible_total.append(
            (
                naive_score_total(
                    {
                        "total": total_amount,
                        "has_adjacent": has_adjacent,
                        "has_currency": has_currency,
                        "sep_decimal": sep_decimal,
                        "index": i,
                    },
                    len(total_matches),
                ),
                total_amount,
            )
        )

    date_scores = sorted(possible_dates, key=lambda x: x[0], reverse=True)
    total_scores = sorted(possible_total, key=lambda x: x[0], reverse=True)

    if verbose:
        print("Date matches:")
        print(date_scores)
        print("Total matches:")
        print(total_scores)

    if len(date_scores) == 0 or len(total_scores) == 0:
        date_scores.append((0, None))
        total_scores.append((0, None))

    date = date_scores[0]
    total = total_scores[0]
    if date[0] < 0.35:
        return [1 - date[0], None], total

    return date, total
