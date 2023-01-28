from datetime import datetime
from unidecode import unidecode
import re

# FR Locale

GOOD_DATE_SEPS = ["-", "/", ".", " "]
GOOD_TIME_SEPS = [":", "h", " "]
GOOD_MINUTE_SEPS = [":", "m", " "]

sep_date = lambda name: "(?P<" + name + ">(?:[^\\n\d]{0,3}date)?[^\d]{1,3})"
sep_time = lambda name: "(?P<" + name + ">(?:[^\\n\d]{0,3}(?:heure|hour))?[^\d]{1,3})"
empty_sep = lambda name, min_sep=0: f"(?P<{name}>[^\\n\d]{{{min_sep},3}})"

day = "(?P<day>[12]\d|3[01]|0[1-9]|[1-9])"
month = "(?P<month>0[1-9]|1[012]|[1-9])"


def current_year_regex():
    current_year = datetime.now().year
    this_decade = (current_year % 100) // 10
    this_year = (current_year % 10)
    
    print("".join(map(str, range(this_year+1))), current_year, this_decade, this_year)
    return f'(?P<year>(?:19)?[789]\d|(?:20)?[{"".join(map(str, range(this_decade)))}]\d|(?:20)?{this_decade}[{"".join(map(str, range(this_year+1)))}])'

year = current_year_regex()
print(year)

hourInside = "[01]\d|2[0-3]"
hour = "(?P<hour>" + hourInside + ")"
minute = "(?P<minute>[0-5]\d)"
second = "(?P<second>[0-5]\d)"

monthNames = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
monthNamesFr = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]
specialMonths = {
    "jun": "juin",
    "jul": "juillet",
}
monthNamesFrNoAccent = list(map(unidecode, monthNamesFr))


def monthShort(month):
    if len(month) == 3:
        return f"(?:{month}\.?)"
    elif len(month) == 4:
        return f"(?:(?:{month[:3]}(?:{month[3]})?)\.?)"
    else:
        return f"(?:(?:{month[:3]}(?:{month[3]}(?:{month[4:]})?)?)\.?)"


naive_date_regex = f"{day}(?:(?:(?P<sep_day>\/|-){month}(?P<sep_month>\/|-))|(?:(?P<sep_day_inner> )(?P<month_name>{'|'.join(list(map(monthShort, monthNamesFrNoAccent)) + list(specialMonths.keys()))})(?P<sep_month_inner> ))){year}(?:[^\/](?!(?:{hourInside})(?:{'|'.join(filter(lambda char: char != ' ', GOOD_TIME_SEPS))})))*[^\d]?(?P<time>{hour}(?P<sep_hour>{'|'.join(filter(lambda char: char != ' ', GOOD_TIME_SEPS))}){minute}(?:(?P<sep_minute>{'|'.join(filter(lambda char: char != ' ', GOOD_MINUTE_SEPS))}){second})?)?"
regex_date_prefix_time = f"(?=(?:(?:(?P<time>{hour}{empty_sep('sep_hour')}{minute}({empty_sep('sep_minute')}{second})??)?{sep_date('sep_date_inner')})|{sep_date('sep_date_outer')}?)?{day}{empty_sep('sep_day')}((?P<month_name>{'|'.join(list(map(monthShort, monthNamesFrNoAccent)) + list(specialMonths.keys()))})|{month}){empty_sep('sep_month')}({year}))"
regex_date_suffix_time = f"(?={day}{empty_sep('sep_day')}((?P<month_name>{'|'.join(list(map(monthShort, monthNamesFrNoAccent)) + list(specialMonths.keys()))})|{month}){empty_sep('sep_month')}({year})(?P<time>{sep_time('sep_time_inner')}{hour}{empty_sep('sep_hour', 1)}{minute}({empty_sep('sep_minute')}{second})?)?)"
regex_date_prefix_time_en_format = f"(?=(?:(?:(?P<time>{hour}{empty_sep('sep_hour')}{minute}({empty_sep('sep_minute')}{second})??)?{sep_date('sep_date_inner')})|{sep_date('sep_date_outer')})?(?P<month_name>{'|'.join(list(map(monthShort, monthNamesFrNoAccent)) + list(specialMonths.keys()))}){empty_sep('sep_month')}{day}{empty_sep('sep_day')}({year}))"
regex_date_suffix_time_en_format = f"(?=(?P<month_name>{'|'.join(list(map(monthShort, monthNamesFrNoAccent)) + list(specialMonths.keys()))}){empty_sep('sep_month')}{day}{empty_sep('sep_day')}({year})(?P<time>{sep_time('sep_time_inner')}{hour}{empty_sep('sep_hour', 1)}{minute}({empty_sep('sep_minute')}{second})?)?)"


def good_seps(sep1, sep2, sep_type):
    if sep_type == "time":
        if sep1 == "h" and sep2 == "m":
            return True

        if sep1 in GOOD_TIME_SEPS and (sep2 != "" and (not sep2 or sep1 == sep2)):
            return True

    if sep1 in GOOD_DATE_SEPS and (sep2 != "" and (not sep2 or sep1 == sep2)):
        return True

    return False


def get_sep_score(sep1, sep2, good_seps):
    if sep1 in good_seps:
        if sep1 == sep2:
            if sep1 != " ":
                return 80
            else:
                return 20

        if sep2 in good_seps:
            return 5
        else:
            return 10

    if sep2 in good_seps:
        return 10

    return 0


def get_digit_score(is_day_one_digit, is_month_one_digit):
    if is_day_one_digit and is_month_one_digit:
        return 20

    if not is_day_one_digit and not is_month_one_digit:
        return 40

    if is_day_one_digit:
        return 10

    return 0


def naive_score_date(possibility, len_matches):
    # Generate score based on whether the date is close to the current date (in an exponential )

    n_days_diff = (datetime.now() - possibility["date"]).days + 1
    score = (1 / (1 + n_days_diff)) * 100
    score += 70 if possibility["has_time"] else 0
    score += (
        120
        if possibility["is_month_name"]
        else (
            get_sep_score(
                possibility["sep_day"], possibility["sep_month"], GOOD_DATE_SEPS
            )
            + get_digit_score(
                possibility["is_day_one_digit"], possibility["is_month_one_digit"]
            )
        )
    )

    score += 20 if not possibility["is_year_two_digit"] else 0

    if possibility["sep_time"]:
        score += (
            50
            if "heure" in possibility["sep_time"] or "hour" in possibility["sep_time"]
            else 0
        )

    if possibility["sep_date"]:
        score += (
            50
            if "heure" in possibility["sep_date"] or "hour" in possibility["sep_date"]
            else 0
        )

    score += (
        get_sep_score(
            possibility["sep_hour"], possibility["sep_minute"], GOOD_TIME_SEPS
        )
        if possibility["has_time"]
        else 0
    )
    # Prioritize small and large indices with two normal distributions
    # Prioritize small indices
    score += (1 / (1 + possibility["index"])) * 15

    # Prioritize large indices
    score += (1 / (1 + len_matches - possibility["index"])) * 25

    return round(score, 2) / 300


def get_month_from_name(name):
    if name in specialMonths:
        name = specialMonths[name]

    possible_months = [
        i + 1 for i, monthName in enumerate(monthNamesFrNoAccent) if name in monthName
    ]
    if len(possible_months) > 0:
        return possible_months[0]

    raise Exception(f"Invalid month name {name}")


def parse_datetime(match):
    is_day_one_digit = len(match.group("day")) == 1
    day = int(match.group("day"))

    if "month" in match.groupdict():
        is_month_name = match.group("month") is None
    else:
        is_month_name = True

    month = (
        6
        if match.group("month_name") == "jun"
        else (
            get_month_from_name(match.group("month_name"))
            if is_month_name
            else int(match.group("month"))
        )
    )

    is_month_one_digit = len(match.group("month")) == 1 if not is_month_name else False
    year = int(match.group("year"))

    is_year_two_digit = False
    if year < 100:
        is_year_two_digit = True
        if year < 70:
            year = 2000 + year
        else:
            year = 1900 + year

    # print(f'Parsed date {year}-{month}-{day} (day one digit: {is_day_one_digit}, month name: {is_month_name})')
    if match.group("time") is not None:
        hours = int(match.group("hour"))
        minutes = int(match.group("minute"))
        seconds = match.group("second") is not None and int(match.group("second")) or 0

        return (
            datetime(year, month, day, hours, minutes, seconds),
            is_year_two_digit,
            is_day_one_digit,
            is_month_one_digit,
            is_month_name,
            True,
        )

    return (
        datetime(year, month, day),
        is_year_two_digit,
        is_day_one_digit,
        is_month_one_digit,
        is_month_name,
        False,
    )


# print(regex_date_prefix_time)
