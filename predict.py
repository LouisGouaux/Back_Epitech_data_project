import datetime
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates

res = JoursFeries.for_year(2026)

d = SchoolHolidayDates()
holidays = d.holidays_for_year(2018)
