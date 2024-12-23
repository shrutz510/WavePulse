import pytz


def get_timezone_by_state(state_code):
    # Dictionary mapping US state codes to their corresponding timezone name strings
    state_timezones = {
        "AL": "US/Central",
        "AK": "US/Alaska",
        "AZ": "US/Arizona",
        "AR": "US/Central",
        "CA": "US/Pacific",
        "CO": "US/Mountain",
        "CT": "US/Eastern",
        "DE": "US/Eastern",
        "DC": "US/Eastern",
        "FL": "US/Eastern",
        "GA": "US/Eastern",
        "HI": "US/Hawaii",
        "ID": "US/Mountain",
        "IL": "US/Central",
        "IN": "US/Eastern",
        "IA": "US/Central",
        "KS": "US/Central",
        "KY": "US/Eastern",
        "LA": "US/Central",
        "ME": "US/Eastern",
        "MD": "US/Eastern",
        "MA": "US/Eastern",
        "MI": "US/Eastern",
        "MN": "US/Central",
        "MS": "US/Central",
        "MO": "US/Central",
        "MT": "US/Mountain",
        "NE": "US/Central",
        "NV": "US/Pacific",
        "NH": "US/Eastern",
        "NJ": "US/Eastern",
        "NM": "US/Mountain",
        "NY": "US/Eastern",
        "NC": "US/Eastern",
        "ND": "US/Central",
        "OH": "US/Eastern",
        "OK": "US/Central",
        "OR": "US/Pacific",
        "PA": "US/Eastern",
        "RI": "US/Eastern",
        "SC": "US/Eastern",
        "SD": "US/Central",
        "TN": "US/Central",
        "TX": "US/Central",
        "UT": "US/Mountain",
        "VT": "US/Eastern",
        "VA": "US/Eastern",
        "WA": "US/Pacific",
        "WV": "US/Eastern",
        "WI": "US/Central",
        "WY": "US/Mountain",
    }

    # Get the timezone name string from the dictionary
    timezone_name = state_timezones.get(state_code.upper())

    if timezone_name:
        return pytz.timezone(timezone_name)
    else:
        return "Invalid state code"


def convert_timezone(current_time, current_state_code, new_state_code):
    current_time_zone = get_timezone_by_state(current_state_code)
    current_time = current_time_zone.localize(current_time)
    new_time_zone = get_timezone_by_state(new_state_code)
    return current_time.astimezone(new_time_zone)


# Driver code
if __name__ == "__main__":
    state_code = "IA"
    timezone = get_timezone_by_state(state_code)
    print(f"The timezone for {state_code} is {timezone}")
