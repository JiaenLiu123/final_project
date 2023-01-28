# from date_regex import regex_date_prefix_time, naive_score_date, parse_datetime
# from date_regex import regex_date_suffix_time, regex_date_prefix_time_en_format, regex_date_suffix_time_en_format
# import re
# from unidecode import unidecode

# # print(regex_date_prefix_time_en_format, '\n\n', regex_date_suffix_time_en_format)
# # print(regex_date_suffix_time_en_format)

from test_all_regex import test_regex

text = """
Carrefour city
CRF-CITY LA ROCHELLE
33 RUE DE LA SCIERIE
17000 LA ROCHELLE
Tel : 05.46.27.02.12
DESCRIPTION QTE MONTANT
150G CHIPS GOURMAN 0.66€
1L PS VELOUTE LEG 1.88€
200G  CHOCOLAT LAIT 1.98€
420G  SAUCE NAPOLIT 1.13€
BEIGNETS CALAMARS 2.34€
BROCOLI FILME 0.99€
CLEMENTINE 1.36€
0.5900kg x 2.30€/kg
EMMENTAL RAPE CRF 1.57€
PET 1.5L EAU MIN. 0.58€
PIZZA TONNO 2.52€
SAC POCKET CARREFO 1.00€
11 ARTICLE (S) TOTAL A PAYER 16.0€
CARTE BANCAIRE EMV EUR 16.0l€
Si vous aviez la carte fidélité,
vous auriez cumulé 0.10€ sur
votre compte fidélité Carrefour,
Détails:
200G CHOCOLAT LAIT 0.10€
0008 003 000416 20/01/2016 19:11:51
MERCI DE VOTRE VISITE
A BIENTOT
"""

test_regex(text, True)

# text = '''Fruits-Tradition
# Tel: 02 38 55 12 18
# 14 05 2017 11 14 2( 1) # 1 T0147
# kg €/k g €
# POMME
# 2.180 1.20 2,62
# SALADE
# 1x 1.10 1.10
# 2 Art. Tot 3.72
# # FREDERIQUE
# Merci de votre visite
# '''

# text1 = unidecode(text).lower()

# # print(regex_date_suffix_time)
# # print()
# date_matches = [(match, i) for i, match in enumerate(re.finditer(regex_date_suffix_time, text1))] + [(match, i) for i, match in enumerate(re.finditer(regex_date_prefix_time, text1))]  #+ [(match, i) for i, match in enumerate(re.finditer(regex_date_prefix_time_en_format, text1))] + [(match, i) for i, match in enumerate(re.finditer(regex_date_suffix_time_en_format, text1))]
# possible_dates = []

# # print(date_matches)
# # print()
# for match, i in date_matches:
#     try:
#         parsed_date, is_day_one_digit, is_month_one_digit, is_month_name, has_time = parse_datetime(match)
#         sep_day = match.group('sep_day').strip()
#         sep_month = match.group('sep_month').strip()

#         sep_hour = match.group('sep_hour').strip() if match.group('sep_hour') else None
#         sep_minute = match.group('sep_minute').strip() if match.group('sep_minute') else None

#         possible_dates.append((naive_score_date({
#             "date": parsed_date,
#             "is_day_one_digit": is_day_one_digit,
#             "is_month_one_digit": is_month_one_digit,
#             "sep_day": sep_day,
#             "sep_month": sep_month,
#             "sep_hour": sep_hour,
#             "sep_minute": sep_minute,
#             "is_month_name": is_month_name,
#             "has_time": has_time,
#             "index": i,
#         }, len(date_matches)), parsed_date))

#     except Exception as e:
#         print('ERR', e)
#         pass # Impossible date found

# print(list(sorted(possible_dates, key=lambda x: x[0])))
