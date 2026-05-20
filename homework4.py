def next_birthday(date, birthdays):

    # Number of days in each month (non-leap year)
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30,
        5: 31, 6: 30, 7: 31, 8: 31,
        9: 30, 10: 31, 11: 30, 12: 31
    }

    # Start from the given date
    month, day = date

    # Check each following day, up to one full year
    for _ in range(365):
        # Move to the next day
        day += 1

        if day > days_in_month[month]:
            day = 1
            month += 1

            # If the month exceeds December, wrap to January
            if month > 12:
                month = 1

        # Current date being checked
        birthday = (month, day)

        # If someone has a birthday on this date, return it
        if birthday in birthdays:
            list_of_names = birthdays[birthday]
            return birthday, list_of_names

    # If no birthdays are found (empty dictionary)
    return (1, 1), []